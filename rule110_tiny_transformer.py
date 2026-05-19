import argparse
import contextlib
import math
import os
import struct
import time
import zlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


def autocast_context(device, dtype_name):
    if device != "cuda" or dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA in this script")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def rule_table_by_code(rule):
    return torch.tensor([(rule >> code) & 1 for code in range(8)], dtype=torch.long)


def eca_step(state, rule_table):
    """One wrapped elementary CA step for a batch of binary 1D states."""
    left = torch.roll(state, shifts=1, dims=-1)
    center = state
    right = torch.roll(state, shifts=-1, dims=-1)
    code = left * 4 + center * 2 + right
    return rule_table[code]


def eca_history(initial_state, frames, rule_table):
    """Return frames consecutive states, including the initial state."""
    states = [initial_state]
    state = initial_state
    for _ in range(frames - 1):
        state = eca_step(state, rule_table)
        states.append(state)
    return torch.stack(states, dim=1)


def soft_eca_step(probabilities, rule_table):
    """Expected ECA next row for independent Bernoulli cell probabilities."""
    left = torch.roll(probabilities, shifts=1, dims=-1)
    center = probabilities
    right = torch.roll(probabilities, shifts=-1, dims=-1)
    expected = torch.zeros_like(probabilities)
    for code in range(8):
        output = float(rule_table[code].item())
        if output == 0.0:
            continue
        left_bit = (code >> 2) & 1
        center_bit = (code >> 1) & 1
        right_bit = code & 1
        term = left if left_bit else (1.0 - left)
        term = term * (center if center_bit else (1.0 - center))
        term = term * (right if right_bit else (1.0 - right))
        expected = expected + output * term
    return expected.clamp(1.0e-6, 1.0 - 1.0e-6)


def make_lm_batch(batch_size, width, frames, direction, rule_table, device):
    initial = torch.randint(0, 2, (batch_size, width), device=device)
    history = eca_history(initial, frames, rule_table)
    if direction == "reverse":
        history = torch.flip(history, dims=(1,))
    seq = history.reshape(batch_size, frames * width)
    x = seq[:, :-1]
    y = seq[:, 1:]
    target_positions = torch.arange(1, frames * width, device=device)
    loss_mask = target_positions >= width
    return x, y, loss_mask.expand(batch_size, -1)


class LMBatchGenerator:
    def __init__(self, width, frames, direction, rule_table, device):
        self.width = width
        self.frames = frames
        self.direction = direction
        self.rule_table = rule_table.to(device=device)
        target_positions = torch.arange(1, frames * width, device=device)
        self.loss_mask = target_positions >= width
        self.device = device

    def __call__(self, batch_size, device):
        initial = torch.randint(0, 2, (batch_size, self.width), device=device)
        history = eca_history(initial, self.frames, self.rule_table)
        if self.direction == "reverse":
            history = torch.flip(history, dims=(1,))
        seq = history.reshape(batch_size, self.frames * self.width)
        return seq[:, :-1], seq[:, 1:], self.loss_mask.expand(batch_size, -1)


def make_transition_batch(batch_size, width, rule_table, device):
    x = torch.randint(0, 2, (batch_size, width), device=device)
    y = eca_step(x, rule_table)
    return x, y, None


class TransitionBatchGenerator:
    def __init__(self, width, rule_table, device):
        self.width = width
        self.rule_table = rule_table.to(device=device)

    def __call__(self, batch_size, device):
        x = torch.randint(0, 2, (batch_size, self.width), device=device)
        y = eca_step(x, self.rule_table)
        return x, y, None


def write_rgb_png(path, pixels):
    height = len(pixels)
    width = len(pixels[0])
    raw_rows = []
    for row in pixels:
        raw_rows.append(b"\x00" + bytes(channel for pixel in row for channel in pixel))
    raw = b"".join(raw_rows)

    def chunk(kind, data):
        payload = kind + data
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", checksum)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def draw_line(pixels, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    x = x0
    y = y0
    while True:
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
            pixels[y][x] = color
        if x == x1 and y == y1:
            break
        error2 = 2 * error
        if error2 >= dy:
            error += dy
            x += sx
        if error2 <= dx:
            error += dx
            y += sy


def render_loss_curve_png(path, curves, width=1000, height=620):
    bg = (255, 255, 255)
    axis = (38, 39, 40)
    grid = (224, 224, 224)
    colors = {
        30: (44, 112, 179),
        110: (219, 84, 59),
    }
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    left = 70
    right = 40
    top = 35
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_points = [point for points in curves.values() for point in points]
    max_step = max(step for step, _ in all_points)
    min_loss = min(loss for _, loss in all_points)
    max_loss = max(loss for _, loss in all_points)
    min_loss = min(0.0, min_loss)
    max_loss = max(math.log(2), max_loss)
    pad = (max_loss - min_loss) * 0.08
    min_loss -= pad
    max_loss += pad

    for i in range(6):
        y = top + round(plot_h * i / 5)
        draw_line(pixels, left, y, width - right, y, grid)
    for i in range(6):
        x = left + round(plot_w * i / 5)
        draw_line(pixels, x, top, x, height - bottom, grid)
    draw_line(pixels, left, top, left, height - bottom, axis)
    draw_line(pixels, left, height - bottom, width - right, height - bottom, axis)

    def project(step, loss):
        x = left + round((step / max_step) * plot_w)
        y = top + round((1.0 - (loss - min_loss) / (max_loss - min_loss)) * plot_h)
        return x, y

    for rule, points in curves.items():
        color = colors.get(rule, (30, 30, 30))
        previous = None
        for step, loss in points:
            current = project(step, loss)
            if previous is not None:
                draw_line(pixels, previous[0], previous[1], current[0], current[1], color)
                draw_line(pixels, previous[0], previous[1] + 1, current[0], current[1] + 1, color)
            previous = current
        legend_x = width - right - 180
        legend_y = top + 25 + 30 * list(curves.keys()).index(rule)
        draw_line(pixels, legend_x, legend_y, legend_x + 45, legend_y, color)

    write_rgb_png(path, pixels)


def render_frame_accuracy_png(path, accuracies, counts, width=1000, height=620):
    bg = (255, 255, 255)
    axis = (38, 39, 40)
    grid = (224, 224, 224)
    line = (44, 112, 179)
    point = (219, 84, 59)
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    left = 70
    right = 40
    top = 35
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    valid = [(idx, acc) for idx, (acc, count) in enumerate(zip(accuracies, counts)) if count > 0]
    if not valid:
        write_rgb_png(path, pixels)
        return
    min_frame = min(idx for idx, _ in valid)
    max_frame = max(idx for idx, _ in valid)
    min_acc = min(acc for _, acc in valid)
    max_acc = max(acc for _, acc in valid)
    min_acc = max(0.0, min_acc - 0.03)
    max_acc = min(1.0, max_acc + 0.03)
    if max_acc <= min_acc:
        max_acc = min(1.0, min_acc + 0.01)

    for i in range(6):
        y = top + round(plot_h * i / 5)
        draw_line(pixels, left, y, width - right, y, grid)
    for i in range(6):
        x = left + round(plot_w * i / 5)
        draw_line(pixels, x, top, x, height - bottom, grid)
    draw_line(pixels, left, top, left, height - bottom, axis)
    draw_line(pixels, left, height - bottom, width - right, height - bottom, axis)

    def project(frame, acc):
        denom = max(1, max_frame - min_frame)
        x = left + round(((frame - min_frame) / denom) * plot_w)
        y = top + round((1.0 - (acc - min_acc) / (max_acc - min_acc)) * plot_h)
        return x, y

    previous = None
    for frame, acc in valid:
        current = project(frame, acc)
        if previous is not None:
            draw_line(pixels, previous[0], previous[1], current[0], current[1], line)
            draw_line(pixels, previous[0], previous[1] + 1, current[0], current[1] + 1, line)
        x, y = current
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dx) + abs(dy) <= 3 and 0 <= y + dy < height and 0 <= x + dx < width:
                    pixels[y + dy][x + dx] = point
        previous = current
    write_rgb_png(path, pixels)


def render_frame_metric_comparison_png(path, panels, width=1600, height=620):
    bg = (255, 255, 255)
    axis = (38, 39, 40)
    grid = (224, 224, 224)
    line_colors = [(44, 112, 179), (219, 84, 59)]
    point = (38, 39, 40)
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    panel_gap = 60
    panel_width = (width - panel_gap) // 2

    valid_values = [
        acc
        for accuracies, counts in panels
        for acc, count in zip(accuracies, counts)
        if count > 0
    ]
    if not valid_values:
        write_rgb_png(path, pixels)
        return
    min_acc = max(0.0, min(valid_values) - 0.03)
    max_acc = min(1.0, max(valid_values) + 0.03)
    if max_acc <= min_acc:
        max_acc = min(1.0, min_acc + 0.01)

    for panel_idx, (accuracies, counts) in enumerate(panels):
        x_offset = panel_idx * (panel_width + panel_gap)
        left = x_offset + 70
        right = x_offset + panel_width - 40
        top = 35
        bottom = height - 70
        plot_w = right - left
        plot_h = bottom - top
        valid = [(idx, acc) for idx, (acc, count) in enumerate(zip(accuracies, counts)) if count > 0]
        if not valid:
            continue
        min_frame = min(idx for idx, _ in valid)
        max_frame = max(idx for idx, _ in valid)

        for i in range(6):
            y = top + round(plot_h * i / 5)
            draw_line(pixels, left, y, right, y, grid)
        for i in range(6):
            x = left + round(plot_w * i / 5)
            draw_line(pixels, x, top, x, bottom, grid)
        draw_line(pixels, left, top, left, bottom, axis)
        draw_line(pixels, left, bottom, right, bottom, axis)

        def project(frame, acc):
            denom = max(1, max_frame - min_frame)
            x = left + round(((frame - min_frame) / denom) * plot_w)
            y = top + round((1.0 - (acc - min_acc) / (max_acc - min_acc)) * plot_h)
            return x, y

        previous = None
        line = line_colors[panel_idx % len(line_colors)]
        for frame, acc in valid:
            current = project(frame, acc)
            if previous is not None:
                draw_line(pixels, previous[0], previous[1], current[0], current[1], line)
                draw_line(pixels, previous[0], previous[1] + 1, current[0], current[1] + 1, line)
            x, y = current
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) + abs(dy) <= 3 and 0 <= y + dy < height and 0 <= x + dx < width:
                        pixels[y + dy][x + dx] = point
            previous = current

    write_rgb_png(path, pixels)


def expand_grid(grid, scale, on_color, off_color):
    pixels = []
    for row in grid.tolist():
        expanded_row = []
        for value in row:
            color = on_color if value else off_color
            expanded_row.extend([color] * scale)
        for _ in range(scale):
            pixels.append(list(expanded_row))
    return pixels


def render_reconstruction_png(path, true_history, reconstructed_history, scale):
    true_history = true_history.cpu().to(torch.long)
    reconstructed_history = reconstructed_history.cpu().to(torch.long)
    errors = true_history != reconstructed_history

    off = (245, 245, 242)
    on = (24, 26, 27)
    ok = (232, 232, 226)
    err = (218, 55, 50)
    gap = [[(255, 255, 255)] * (true_history.shape[1] * scale) for _ in range(scale)]

    true_panel = expand_grid(true_history, scale, on, off)
    reconstructed_panel = expand_grid(reconstructed_history, scale, on, off)
    error_panel = expand_grid(errors.to(torch.long), scale, err, ok)
    pixels = true_panel + gap + reconstructed_panel + gap + error_panel
    write_rgb_png(path, pixels)


def render_attention_png(path, attention_grid, scale):
    attention_grid = attention_grid.cpu().float()
    max_value = attention_grid.max().item()
    if max_value <= 0:
        max_value = 1.0

    pixels = []
    for row in attention_grid.tolist():
        expanded_row = []
        for value in row:
            intensity = math.sqrt(max(0.0, value) / max_value)
            red = round(255 - 28 * intensity)
            green = round(255 - 175 * intensity)
            blue = round(255 - 210 * intensity)
            expanded_row.extend([(red, green, blue)] * scale)
        for _ in range(scale):
            pixels.append(list(expanded_row))
    write_rgb_png(path, pixels)


def render_attention_overlay_png(path, history, attention_grid, scale):
    history = history.cpu().to(torch.long)
    attention_grid = attention_grid.cpu().float()
    max_value = attention_grid.max().item()
    if max_value <= 0:
        max_value = 1.0

    off = (245, 245, 242)
    on = (24, 26, 27)
    low = (47, 115, 255)
    high = (235, 51, 55)
    alpha = 0.62
    pixels = []
    for history_row, attention_row in zip(history.tolist(), attention_grid.tolist()):
        expanded_row = []
        for cell, value in zip(history_row, attention_row):
            t = math.sqrt(max(0.0, value) / max_value)
            tint = tuple(round(low[i] * (1.0 - t) + high[i] * t) for i in range(3))
            base = on if cell else off
            color = tuple(round(base[i] * (1.0 - alpha) + tint[i] * alpha) for i in range(3))
            expanded_row.extend([color] * scale)
        for _ in range(scale):
            pixels.append(list(expanded_row))
    write_rgb_png(path, pixels)


def render_history_png(path, history, scale):
    off = (245, 245, 242)
    on = (24, 26, 27)
    pixels = expand_grid(history.cpu().to(torch.long), scale, on, off)
    write_rgb_png(path, pixels)


def parse_attention_targets(value, width, frames):
    if not value:
        defaults = [1, frames // 4, frames // 2, frames - 1]
        seen = set()
        targets = []
        for frame in defaults:
            frame = max(1, min(frames - 1, frame))
            target = (frame, width // 2)
            if target not in seen:
                targets.append(target)
                seen.add(target)
        return targets

    targets = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--attention-targets entries must be frame:cell")
        frame_text, cell_text = part.split(":", 1)
        frame = int(frame_text)
        cell = int(cell_text)
        if not (0 <= frame < frames and 0 <= cell < width):
            raise ValueError(f"attention target out of bounds: {part}")
        targets.append((frame, cell))
    return targets


@torch.no_grad()
def reconstruct_reverse(model, history, device):
    """Generate a reversed history from the final row, then return forward-time rows."""
    model.eval()
    reversed_history = torch.flip(history, dims=(1,))
    generated = reversed_history[:, 0].reshape(history.shape[0], history.shape[2])
    total_tokens = history.shape[1] * history.shape[2]
    while generated.shape[1] < total_tokens:
        logits, _ = model(generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    generated_reversed = generated.reshape(history.shape[0], history.shape[1], history.shape[2])
    return torch.flip(generated_reversed, dims=(1,)).to(device)


@torch.no_grad()
def reconstruct_reverse_autoregressive(model, history, device, dtype_name):
    """Autoregressively generate reverse-time tokens from only the final natural row."""
    model.eval()
    reversed_history = torch.flip(history, dims=(1,))
    generated = reversed_history[:, 0].reshape(history.shape[0], history.shape[2])
    total_tokens = history.shape[1] * history.shape[2]
    while generated.shape[1] < total_tokens:
        with autocast_context(device, dtype_name):
            logits, _ = model(generated)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    generated_reversed = generated.reshape(history.shape[0], history.shape[1], history.shape[2])
    return torch.flip(generated_reversed, dims=(1,)).to(device)


def tiny_gpt_cached_step(model, token, position, caches):
    if not isinstance(model, TinyGPT):
        raise ValueError("cached autoregressive generation currently requires TinyGPT")
    positions = model.positions[position : position + 1]
    x = model.token_emb(token[:, None])
    if model.position_encoding in {"sequential", "both"}:
        x = x + model.pos_emb(positions)[None, :, :]
    if model.position_encoding in {"grid", "both"}:
        x = x + model.frame_emb(model.frame_positions[position : position + 1])[None, :, :]
        x = x + model.cell_emb(model.cell_positions[position : position + 1])[None, :, :]
        x = x + model.target_frame_emb(model.target_frame_positions[position : position + 1])[None, :, :]
        x = x + model.target_cell_emb(model.target_cell_positions[position : position + 1])[None, :, :]

    next_caches = []
    for layer_idx, block in enumerate(model.blocks):
        h = block.ln1(x)
        projection = F.linear(h, block.attn.in_proj_weight, block.attn.in_proj_bias)
        q, k, v = projection.chunk(3, dim=-1)
        batch_size, _, embed_dim = q.shape
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads
        q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        if caches[layer_idx] is not None:
            previous_k, previous_v = caches[layer_idx]
            k = torch.cat([previous_k, k], dim=2)
            v = torch.cat([previous_v, v], dim=2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)
        attention_output = block.attn.out_proj(attention_output)
        x = x + attention_output
        x = x + block.mlp(block.ln2(x))
        next_caches.append((k, v))

    logits = model.head(model.ln(x))[:, 0]
    return logits, next_caches


@torch.no_grad()
def reconstruct_reverse_autoregressive_cached(model, history, device, dtype_name):
    """Cached autoregressive reverse generation for the unmasked causal TinyGPT."""
    model.eval()
    reversed_history = torch.flip(history, dims=(1,))
    generated = reversed_history[:, 0].reshape(history.shape[0], history.shape[2])
    total_tokens = history.shape[1] * history.shape[2]
    caches = [None for _ in model.blocks]
    next_position = 0
    logits = None
    with autocast_context(device, dtype_name):
        while next_position < generated.shape[1]:
            logits, caches = tiny_gpt_cached_step(
                model,
                generated[:, next_position],
                next_position,
                caches,
            )
            next_position += 1
        while generated.shape[1] < total_tokens:
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.shape[1] == total_tokens:
                break
            logits, caches = tiny_gpt_cached_step(
                model,
                next_token[:, 0],
                next_position,
                caches,
            )
            next_position += 1
    generated_reversed = generated.reshape(history.shape[0], history.shape[1], history.shape[2])
    return torch.flip(generated_reversed, dims=(1,)).to(device)


@torch.no_grad()
def predict_natural_row_autoregressive(model, history, natural_frame, device, dtype_name):
    """Generate one natural row while keeping all later natural rows teacher-forced."""
    if not isinstance(model, TinyGPT):
        raise ValueError("one-row autoregressive eval currently requires TinyGPT")
    if not (0 <= natural_frame < history.shape[1] - 1):
        raise ValueError("natural_frame must have a later conditioning row")
    model.eval()
    model_history = torch.flip(history, dims=(1,))
    model_frame = history.shape[1] - 1 - natural_frame
    prefix_tokens = model_history[:, :model_frame].reshape(history.shape[0], model_frame * history.shape[2])
    caches = [None for _ in model.blocks]
    next_position = 0
    logits = None
    with autocast_context(device, dtype_name):
        while next_position < prefix_tokens.shape[1]:
            logits, caches = tiny_gpt_cached_step(
                model,
                prefix_tokens[:, next_position],
                next_position,
                caches,
            )
            next_position += 1
        generated_row = []
        for cell_idx in range(history.shape[2]):
            next_token = logits.argmax(dim=-1)
            generated_row.append(next_token)
            if cell_idx == history.shape[2] - 1:
                break
            logits, caches = tiny_gpt_cached_step(model, next_token, next_position, caches)
            next_position += 1
    return torch.stack(generated_row, dim=1)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class RowMaskedBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.lnq = nn.LayerNorm(d_model)
        self.lnkv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, context, attn_mask, return_attn=False):
        hq = self.lnq(query)
        hkv = self.lnkv(context)
        attn_out, attn_weights = self.attn(
            hq,
            hkv,
            hkv,
            attn_mask=attn_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        query = query + attn_out
        query = query + self.mlp(self.ln2(query))
        if return_attn:
            return query, attn_weights
        return query


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_len,
        d_model,
        n_heads,
        n_layers,
        dropout,
        width=None,
        frames=None,
        position_encoding="grid",
    ):
        super().__init__()
        if position_encoding not in {"grid", "sequential", "both"}:
            raise ValueError("position_encoding must be grid, sequential, or both")
        if position_encoding in {"grid", "both"} and (width is None or frames is None):
            raise ValueError("grid position encoding requires width and frames")

        self.width = width
        self.frames = frames
        self.position_encoding = position_encoding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if position_encoding in {"sequential", "both"}:
            self.pos_emb = nn.Embedding(context_len, d_model)
        if position_encoding in {"grid", "both"}:
            self.frame_emb = nn.Embedding(frames, d_model)
            self.cell_emb = nn.Embedding(width, d_model)
            self.target_frame_emb = nn.Embedding(frames, d_model)
            self.target_cell_emb = nn.Embedding(width, d_model)
        positions = torch.arange(context_len)
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        if position_encoding in {"grid", "both"}:
            target_positions = positions + 1
            self.register_buffer(
                "frame_positions",
                torch.div(positions, width, rounding_mode="floor"),
                persistent=False,
            )
            self.register_buffer("cell_positions", positions % width, persistent=False)
            self.register_buffer(
                "target_frame_positions",
                torch.div(target_positions, width, rounding_mode="floor"),
                persistent=False,
            )
            self.register_buffer("target_cell_positions", target_positions % width, persistent=False)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, loss_mask=None, return_attn=False):
        _, seq_len = idx.shape
        positions = self.positions[:seq_len]
        x = self.token_emb(idx)
        if self.position_encoding in {"sequential", "both"}:
            x = x + self.pos_emb(positions)[None, :, :]
        if self.position_encoding in {"grid", "both"}:
            x = x + self.frame_emb(self.frame_positions[:seq_len])[None, :, :]
            x = x + self.cell_emb(self.cell_positions[:seq_len])[None, :, :]
            x = x + self.target_frame_emb(self.target_frame_positions[:seq_len])[None, :, :]
            x = x + self.target_cell_emb(self.target_cell_positions[:seq_len])[None, :, :]
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, causal_mask)

        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            if loss_mask is not None:
                token_loss = token_loss.reshape_as(targets)
                loss = (token_loss * loss_mask).sum() / loss_mask.sum()
            else:
                loss = token_loss.mean()
        return logits, loss


class TinyRowMaskedGPT(nn.Module):
    def __init__(self, vocab_size, context_len, d_model, n_heads, n_layers, dropout, width, frames):
        super().__init__()
        self.width = width
        self.frames = frames
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.source_frame_emb = nn.Embedding(frames, d_model)
        self.source_cell_emb = nn.Embedding(width, d_model)
        self.target_frame_emb = nn.Embedding(frames, d_model)
        self.target_cell_emb = nn.Embedding(width, d_model)
        source_positions = torch.arange(context_len)
        target_positions = source_positions + 1
        source_frames = torch.div(source_positions, width, rounding_mode="floor")
        target_frames = torch.div(target_positions, width, rounding_mode="floor")
        self.register_buffer("source_frames", source_frames, persistent=False)
        self.register_buffer("source_cells", source_positions % width, persistent=False)
        self.register_buffer("target_frames", target_frames, persistent=False)
        self.register_buffer("target_cells", target_positions % width, persistent=False)
        allow_prior_rows = source_frames[None, :] < target_frames[:, None]
        allow_initial_row = (target_frames[:, None] == 0) & (
            source_positions[None, :] < target_positions[:, None]
        )
        self.register_buffer("attn_mask", ~(allow_prior_rows | allow_initial_row), persistent=False)
        self.blocks = nn.ModuleList(
            [RowMaskedBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, loss_mask=None, return_attn=False):
        _, seq_len = idx.shape
        source_frames = self.source_frames[:seq_len]
        source_cells = self.source_cells[:seq_len]
        target_frames = self.target_frames[:seq_len]
        target_cells = self.target_cells[:seq_len]

        context = self.token_emb(idx)
        context = context + self.source_frame_emb(source_frames)[None, :, :]
        context = context + self.source_cell_emb(source_cells)[None, :, :]

        query = self.target_frame_emb(target_frames)[None, :, :]
        query = query + self.target_cell_emb(target_cells)[None, :, :]
        query = query.expand(idx.shape[0], -1, -1)
        attn_mask = self.attn_mask[:seq_len, :seq_len]

        attentions = []
        for block in self.blocks:
            if return_attn:
                query, attn_weights = block(query, context, attn_mask, return_attn=True)
                attentions.append(attn_weights)
            else:
                query = block(query, context, attn_mask)

        logits = self.head(self.ln(query))
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            if loss_mask is not None:
                token_loss = token_loss.reshape_as(targets)
                loss = (token_loss * loss_mask).sum() / loss_mask.sum()
            else:
                loss = token_loss.mean()
        if return_attn:
            return logits, loss, attentions
        return logits, loss


class TinyTransitionTransformer(nn.Module):
    def __init__(self, vocab_size, width, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(width, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, loss_mask=None):
        _, width = idx.shape
        positions = torch.arange(width, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            if loss_mask is not None:
                token_loss = token_loss.reshape_as(targets)
                loss = (token_loss * loss_mask).sum() / loss_mask.sum()
            else:
                loss = token_loss.mean()
        return logits, loss


@torch.no_grad()
def estimate_loss(model, make_batch, batch_size, eval_batches, device, dtype_name):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    for _ in range(eval_batches):
        x, y, loss_mask = make_batch(batch_size, device)
        with autocast_context(device, dtype_name):
            logits, loss = model(x, y, loss_mask)
        pred = logits.argmax(dim=-1)
        loss_sum += loss.item()
        correct = (pred == y).float()
        if loss_mask is not None:
            correct = correct * loss_mask
            acc_sum += (correct.sum() / loss_mask.sum()).item()
        else:
            acc_sum += correct.mean().item()
        count += 1
    if is_distributed():
        stats = torch.tensor([loss_sum, acc_sum, count], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, acc_sum, count = stats.tolist()
    model.train()
    return loss_sum / count, acc_sum / count


def row_consistency_loss_from_logits(logits, targets, args, rule_table):
    if args.task != "lm" or args.lm_direction != "reverse" or not args.mask_row_prefix:
        return logits.new_tensor(0.0)
    batch_size = logits.shape[0]
    probabilities = logits.softmax(dim=-1)[..., 1]
    predicted_history = torch.zeros(
        batch_size,
        args.frames,
        args.width,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    true_history = torch.zeros(
        batch_size,
        args.frames,
        args.width,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    target_positions = torch.arange(1, args.frames * args.width, device=probabilities.device)
    model_frames = torch.div(target_positions, args.width, rounding_mode="floor")
    natural_frames = args.frames - 1 - model_frames
    cells = target_positions % args.width
    valid = target_positions >= args.width
    predicted_history[:, natural_frames[valid], cells[valid]] = probabilities[:, valid]
    true_history[:, natural_frames[valid], cells[valid]] = targets[:, valid].to(probabilities.dtype)

    losses = []
    for frame in range(args.frames - 1):
        predicted_next = soft_eca_step(predicted_history[:, frame], rule_table)
        true_next = true_history[:, frame + 1]
        losses.append(F.binary_cross_entropy(predicted_next, true_next))
    return torch.stack(losses).mean()


def parse_rule_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_experiment(args, rule, device):
    rule_table = rule_table_by_code(rule).to(device=device)
    if args.task == "lm":
        context_len = args.width * args.frames - 1
        make_batch = LMBatchGenerator(
            width=args.width,
            frames=args.frames,
            direction=args.lm_direction,
            rule_table=rule_table,
            device=device,
        )
        if args.mask_row_prefix:
            if args.position_encoding != "grid":
                raise ValueError("--mask-row-prefix currently requires --position-encoding grid")
            model = TinyRowMaskedGPT(
                vocab_size=2,
                context_len=context_len,
                d_model=args.d_model,
                n_heads=args.heads,
                n_layers=args.layers,
                dropout=args.dropout,
                width=args.width,
                frames=args.frames,
            ).to(device)
        else:
            model = TinyGPT(
                vocab_size=2,
                context_len=context_len,
                d_model=args.d_model,
                n_heads=args.heads,
                n_layers=args.layers,
                dropout=args.dropout,
                width=args.width,
                frames=args.frames,
                position_encoding=args.position_encoding,
            ).to(device)
    else:
        context_len = args.width
        make_batch = TransitionBatchGenerator(
            width=args.width,
            rule_table=rule_table,
            device=device,
        )
        model = TinyTransitionTransformer(
            vocab_size=2,
            width=args.width,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    return rule_table, context_len, make_batch, model


def save_training_checkpoint(path, model, args, rule, curve, step, eval_loss=None, eval_acc=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "rule": rule,
            "curve": curve,
            "step": step,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        },
        path,
    )
    print(
        f"wrote_checkpoint={path} "
        f"step={step} "
        f"eval_loss={eval_loss if eval_loss is not None else float('nan'):.4f} "
        f"eval_acc={eval_acc if eval_acc is not None else float('nan'):.3f}",
        flush=True,
    )


def train_experiment(args, rule, device):
    rule_table, context_len, make_batch, model = build_experiment(args, rule, device)
    if args.checkpoint_in is not None:
        checkpoint = torch.load(args.checkpoint_in, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if is_main_process(args):
            print(f"loaded_checkpoint={args.checkpoint_in}", flush=True)
    if args.compile:
        model = torch.compile(model)
    if getattr(args, "distributed", False):
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    parameter_count = sum(p.numel() for p in unwrap_model(model).parameters())
    chance_loss = math.log(2)
    if is_main_process(args):
        print(
            f"rule={rule} task={args.task} lm_direction={args.lm_direction} "
            f"mask_row_prefix={args.mask_row_prefix} "
            f"device={device} dtype={args.dtype} compile={args.compile} "
            f"distributed={getattr(args, 'distributed', False)} "
            f"rank={getattr(args, 'rank', 0)} world_size={getattr(args, 'world_size', 1)} "
            f"per_gpu_batch_size={args.batch_size} "
            f"global_batch_size={args.batch_size * getattr(args, 'world_size', 1)} "
            f"parameters={parameter_count:,} context_len={context_len}",
            flush=True,
        )
        print(f"chance_loss={chance_loss:.4f}", flush=True)

    torch.manual_seed(args.seed + getattr(args, "rank", 0) * 1_000_003)
    curve = []
    best_eval_loss = None
    best_eval_acc = None
    best_step = 0
    start = time.time()
    train_elapsed = 0.0
    checkpoint_end = start
    model.train()
    for step in range(1, args.steps + 1):
        x, y, loss_mask = make_batch(args.batch_size, device)
        with autocast_context(device, args.dtype):
            logits, loss = model(x, y, loss_mask)
            consistency_loss = row_consistency_loss_from_logits(logits, y, args, rule_table)
            loss = loss + args.consistency_loss_weight * consistency_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            if device == "cuda":
                torch.cuda.synchronize()
            before_eval = time.time()
            train_elapsed += before_eval - checkpoint_end
            eval_loss, eval_acc = estimate_loss(
                model,
                make_batch,
                args.batch_size,
                args.eval_batches,
                device,
                args.dtype,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            curve.append((step, eval_loss))
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_acc = eval_acc
                best_step = step
                if args.best_checkpoint_out is not None and is_main_process(args):
                    save_training_checkpoint(
                        args.best_checkpoint_out,
                        unwrap_model(model),
                        args,
                        rule,
                        curve,
                        step,
                        eval_loss,
                        eval_acc,
                    )
            if (
                args.checkpoint_every > 0
                and args.checkpoint_dir is not None
                and (step % args.checkpoint_every == 0 or step == args.steps)
            ):
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f"rule{rule:03d}_step{step:06d}.pt",
                )
                if is_main_process(args):
                    save_training_checkpoint(
                        checkpoint_path,
                        unwrap_model(model),
                        args,
                        rule,
                        curve,
                        step,
                        eval_loss,
                        eval_acc,
                    )
            elapsed = time.time() - start
            checkpoint_end = time.time()
            train_steps_per_sec = step / train_elapsed if train_elapsed > 0 else 0.0
            train_samples_per_sec = train_steps_per_sec * args.batch_size * getattr(args, "world_size", 1)
            if is_main_process(args):
                print(
                    f"rule={rule} "
                    f"step={step:5d} "
                    f"train_loss={loss.item():.4f} "
                    f"consistency_loss={consistency_loss.item():.4f} "
                    f"eval_loss={eval_loss:.4f} "
                    f"eval_acc={eval_acc:.3f} "
                    f"elapsed={elapsed:.1f}s "
                    f"train_elapsed={train_elapsed:.1f}s "
                    f"train_steps_per_sec={train_steps_per_sec:.2f} "
                    f"train_samples_per_sec={train_samples_per_sec:.0f}",
                    flush=True,
                )
    if best_eval_loss is not None and is_main_process(args):
        print(
            f"best_checkpoint_metric=eval_loss "
            f"best_step={best_step} "
            f"best_eval_loss={best_eval_loss:.4f} "
            f"best_eval_acc={best_eval_acc:.3f}",
            flush=True,
        )
    if is_distributed():
        dist.barrier()
    return rule_table, make_batch, unwrap_model(model), curve


@torch.no_grad()
def write_attention_images(model, rule_table, args, device):
    if args.task != "lm" or not args.mask_row_prefix:
        raise ValueError("attention images currently require --task lm --mask-row-prefix")
    model.eval()
    initial = torch.randint(0, 2, (1, args.width), device=device)
    true_history = eca_history(initial, args.frames, rule_table)
    model_history = true_history
    if args.lm_direction == "reverse":
        model_history = torch.flip(true_history, dims=(1,))
    seq = model_history.reshape(1, args.frames * args.width)
    x = seq[:, :-1]
    y = seq[:, 1:]
    target_positions = torch.arange(1, args.frames * args.width, device=device)
    loss_mask = (target_positions >= args.width).expand(1, -1)
    logits, _, attentions = model(x, y, loss_mask, return_attn=True)
    pred = logits.argmax(dim=-1)

    if args.evolution_image is not None:
        render_history_png(args.evolution_image, true_history[0], args.image_scale)
        print(f"wrote_evolution_image={args.evolution_image}", flush=True)

    root, ext = os.path.splitext(args.attention_image)
    if not ext:
        ext = ".png"

    for frame, cell in parse_attention_targets(args.attention_targets, args.width, args.frames):
        natural_frame = frame
        model_frame = frame
        if args.attention_order == "natural" and args.lm_direction == "reverse":
            model_frame = args.frames - 1 - frame
        target_position = model_frame * args.width + cell
        target_index = target_position - 1
        if not (0 <= target_index < y.shape[1]):
            raise ValueError(f"attention target is not predicted: {frame}:{cell}")

        source_positions = torch.arange(x.shape[1], device=device)
        source_frames = torch.div(source_positions, args.width, rounding_mode="floor")
        if args.attention_order == "natural" and args.lm_direction == "reverse":
            source_frames = args.frames - 1 - source_frames
        source_cells = source_positions % args.width

        layer_head_maps = []
        if args.attention_all_heads:
            for layer_idx, layer_attention in enumerate(attentions):
                if args.attention_layer is not None and layer_idx != args.attention_layer:
                    continue
                for head_idx in range(layer_attention.shape[1]):
                    if args.attention_head is not None and head_idx != args.attention_head:
                        continue
                    layer_head_maps.append(
                        (layer_idx, head_idx, layer_attention[0, head_idx, target_index])
                    )
        else:
            layer_head_maps.append((len(attentions) - 1, -1, attentions[-1][0].mean(dim=0)[target_index]))

        for layer_idx, head_idx, source_attention in layer_head_maps:
            attention_grid = torch.zeros(args.frames, args.width, device=device)
            attention_grid[source_frames, source_cells] = source_attention
            head_label = "avg" if head_idx < 0 else f"h{head_idx:02d}"
            path = f"{root}_f{natural_frame:03d}_c{cell:03d}_l{layer_idx:02d}_{head_label}{ext}"
            if args.attention_overlay:
                render_attention_overlay_png(path, true_history[0], attention_grid, args.image_scale)
            else:
                render_attention_png(path, attention_grid, args.image_scale)
            print(
                f"wrote_attention_image={path} "
                f"target_frame={natural_frame} target_cell={cell} "
                f"model_frame={model_frame} layer={layer_idx} head={head_idx} "
                f"true={int(y[0, target_index].item())} "
                f"pred={int(pred[0, target_index].item())}",
                flush=True,
            )

    if args.attention_average_initial_row:
        source_positions = torch.arange(x.shape[1], device=device)
        source_frames = torch.div(source_positions, args.width, rounding_mode="floor")
        if args.attention_order == "natural" and args.lm_direction == "reverse":
            source_frames = args.frames - 1 - source_frames
        source_cells = source_positions % args.width

        model_frame = 0
        if args.attention_order == "natural" and args.lm_direction == "reverse":
            model_frame = args.frames - 1
        target_indices = model_frame * args.width + torch.arange(args.width, device=device) - 1
        if (target_indices < 0).any() or (target_indices >= y.shape[1]).any():
            raise ValueError("initial-row attention average includes an unpredicted target")

        for layer_idx, layer_attention in enumerate(attentions):
            for head_idx in range(layer_attention.shape[1]):
                mean_source_attention = layer_attention[0, head_idx, target_indices].mean(dim=0)
                attention_grid = torch.zeros(args.frames, args.width, device=device)
                attention_grid[source_frames, source_cells] = mean_source_attention
                path = f"{root}_icavg_l{layer_idx:02d}_h{head_idx:02d}{ext}"
                if args.attention_overlay:
                    render_attention_overlay_png(path, true_history[0], attention_grid, args.image_scale)
                else:
                    render_attention_png(path, attention_grid, args.image_scale)
                acc = (pred[0, target_indices] == y[0, target_indices]).float().mean().item()
                print(
                    f"wrote_attention_image={path} "
                    f"target_frame=0 target_cells=all "
                    f"model_frame={model_frame} layer={layer_idx} head={head_idx} "
                    f"target_acc={acc:.3f}",
                    flush=True,
                )


@torch.no_grad()
def evaluate_frame_accuracy(model, rule_table, args, device):
    if args.task != "lm":
        raise ValueError("frame accuracy currently requires --task lm")
    model.eval()
    correct_by_frame = torch.zeros(args.frames, device=device)
    count_by_frame = torch.zeros(args.frames, device=device)
    processed = 0
    start = time.time()

    target_positions = torch.arange(1, args.frames * args.width, device=device)
    target_model_frames = torch.div(target_positions, args.width, rounding_mode="floor")
    valid_targets = target_positions >= args.width
    if args.lm_direction == "reverse":
        target_natural_frames = args.frames - 1 - target_model_frames
    else:
        target_natural_frames = target_model_frames
    target_cells = target_positions % args.width

    while processed < args.frame_accuracy_samples:
        batch_size = min(args.frame_accuracy_batch_size, args.frame_accuracy_samples - processed)
        initial = torch.randint(0, 2, (batch_size, args.width), device=device)
        history = eca_history(initial, args.frames, rule_table)
        model_history = history
        if args.lm_direction == "reverse":
            model_history = torch.flip(history, dims=(1,))
        seq = model_history.reshape(batch_size, args.frames * args.width)
        x = seq[:, :-1]
        y = seq[:, 1:]
        loss_mask = valid_targets
        with autocast_context(device, args.dtype):
            logits, _ = model(x, y, loss_mask.expand(batch_size, -1))
        pred = logits.argmax(dim=-1)
        correct = (pred == y)

        for frame in range(args.frames):
            frame_mask = (target_natural_frames == frame) & valid_targets
            if frame_mask.any():
                frame_correct = correct[:, frame_mask].sum()
                correct_by_frame[frame] += frame_correct
                count_by_frame[frame] += batch_size * frame_mask.sum()
        processed += batch_size
        if processed == args.frame_accuracy_samples or processed % (args.frame_accuracy_batch_size * 10) == 0:
            elapsed = time.time() - start
            print(
                f"frame_accuracy_processed={processed} "
                f"elapsed={elapsed:.1f}s "
                f"samples_per_sec={processed / elapsed if elapsed > 0 else 0.0:.1f}",
                flush=True,
            )

    correct_cpu = correct_by_frame.cpu()
    count_cpu = count_by_frame.cpu()
    accuracies = [
        (correct_cpu[idx] / count_cpu[idx]).item() if count_cpu[idx].item() > 0 else float("nan")
        for idx in range(args.frames)
    ]
    counts = [int(count_cpu[idx].item()) for idx in range(args.frames)]

    if args.frame_accuracy_csv is not None:
        with open(args.frame_accuracy_csv, "w", encoding="utf-8") as f:
            f.write("frame,accuracy,count\n")
            for frame, (accuracy, count) in enumerate(zip(accuracies, counts)):
                accuracy_text = "" if count == 0 else f"{accuracy:.8f}"
                f.write(f"{frame},{accuracy_text},{count}\n")
        print(f"wrote_frame_accuracy_csv={args.frame_accuracy_csv}", flush=True)

    if args.frame_accuracy_image is not None:
        render_frame_accuracy_png(args.frame_accuracy_image, accuracies, counts)
        print(f"wrote_frame_accuracy_image={args.frame_accuracy_image}", flush=True)

    valid_correct = correct_cpu[count_cpu > 0].sum().item()
    valid_count = count_cpu[count_cpu > 0].sum().item()
    print(
        f"frame_accuracy_overall={valid_correct / valid_count if valid_count else 0.0:.4f} "
        f"predicted_frames={int((count_cpu > 0).sum().item())} "
        f"samples={args.frame_accuracy_samples}",
        flush=True,
    )
    return accuracies, counts


@torch.no_grad()
def evaluate_transition_consistency(model, rule_table, args, device):
    if args.task != "lm" or args.lm_direction != "reverse":
        raise ValueError("transition consistency currently requires --task lm --lm-direction reverse")
    model.eval()
    correct_by_frame = torch.zeros(args.frames, device=device)
    count_by_frame = torch.zeros(args.frames, device=device)
    exact_by_frame = torch.zeros(args.frames, device=device)
    exact_count_by_frame = torch.zeros(args.frames, device=device)
    processed = 0
    start = time.time()

    target_positions = torch.arange(1, args.frames * args.width, device=device)
    target_model_frames = torch.div(target_positions, args.width, rounding_mode="floor")
    valid_targets = target_positions >= args.width
    target_natural_frames = args.frames - 1 - target_model_frames
    target_cells = target_positions % args.width
    valid_indices = torch.nonzero(valid_targets, as_tuple=False).flatten()

    while processed < args.transition_consistency_samples:
        batch_size = min(
            args.transition_consistency_batch_size,
            args.transition_consistency_samples - processed,
        )
        initial = torch.randint(0, 2, (batch_size, args.width), device=device)
        history = eca_history(initial, args.frames, rule_table)
        model_history = torch.flip(history, dims=(1,))
        seq = model_history.reshape(batch_size, args.frames * args.width)
        x = seq[:, :-1]
        y = seq[:, 1:]
        with autocast_context(device, args.dtype):
            logits, _ = model(x, y, valid_targets.expand(batch_size, -1))
        pred = logits.argmax(dim=-1)

        predicted_rows = torch.zeros(
            batch_size,
            args.frames,
            args.width,
            dtype=torch.long,
            device=device,
        )
        for idx in valid_indices.tolist():
            frame = int(target_natural_frames[idx].item())
            cell = int(target_cells[idx].item())
            predicted_rows[:, frame, cell] = pred[:, idx]

        for frame in range(args.frames - 1):
            produced_next = eca_step(predicted_rows[:, frame], rule_table)
            correct = produced_next == history[:, frame + 1]
            correct_by_frame[frame] += correct.sum()
            count_by_frame[frame] += correct.numel()
            exact_by_frame[frame] += correct.all(dim=-1).sum()
            exact_count_by_frame[frame] += batch_size

        processed += batch_size
        if (
            processed == args.transition_consistency_samples
            or processed % (args.transition_consistency_batch_size * 10) == 0
        ):
            elapsed = time.time() - start
            print(
                f"transition_consistency_processed={processed} "
                f"elapsed={elapsed:.1f}s "
                f"samples_per_sec={processed / elapsed if elapsed > 0 else 0.0:.1f}",
                flush=True,
            )

    correct_cpu = correct_by_frame.cpu()
    count_cpu = count_by_frame.cpu()
    exact_cpu = exact_by_frame.cpu()
    exact_count_cpu = exact_count_by_frame.cpu()
    accuracies = [
        (correct_cpu[idx] / count_cpu[idx]).item() if count_cpu[idx].item() > 0 else float("nan")
        for idx in range(args.frames)
    ]
    exact_rates = [
        (exact_cpu[idx] / exact_count_cpu[idx]).item()
        if exact_count_cpu[idx].item() > 0
        else float("nan")
        for idx in range(args.frames)
    ]
    counts = [int(count_cpu[idx].item()) for idx in range(args.frames)]
    exact_counts = [int(exact_count_cpu[idx].item()) for idx in range(args.frames)]

    if args.transition_consistency_csv is not None:
        with open(args.transition_consistency_csv, "w", encoding="utf-8") as f:
            f.write("frame,next_cell_accuracy,next_cell_count,next_row_exact_rate,next_row_count\n")
            for frame, (accuracy, count, exact_rate, exact_count) in enumerate(
                zip(accuracies, counts, exact_rates, exact_counts)
            ):
                accuracy_text = "" if count == 0 else f"{accuracy:.8f}"
                exact_text = "" if exact_count == 0 else f"{exact_rate:.8f}"
                f.write(f"{frame},{accuracy_text},{count},{exact_text},{exact_count}\n")
        print(f"wrote_transition_consistency_csv={args.transition_consistency_csv}", flush=True)

    if args.transition_consistency_image is not None:
        render_frame_accuracy_png(args.transition_consistency_image, accuracies, counts)
        print(f"wrote_transition_consistency_image={args.transition_consistency_image}", flush=True)

    valid_correct = correct_cpu[count_cpu > 0].sum().item()
    valid_count = count_cpu[count_cpu > 0].sum().item()
    valid_exact = exact_cpu[exact_count_cpu > 0].sum().item()
    valid_exact_count = exact_count_cpu[exact_count_cpu > 0].sum().item()
    print(
        f"transition_consistency_overall_cell={valid_correct / valid_count if valid_count else 0.0:.4f} "
        f"transition_consistency_overall_row_exact={valid_exact / valid_exact_count if valid_exact_count else 0.0:.4f} "
        f"predicted_source_frames={int((count_cpu > 0).sum().item())} "
        f"samples={args.transition_consistency_samples}",
        flush=True,
    )
    return accuracies, counts, exact_rates, exact_counts


@torch.no_grad()
def evaluate_autoregressive_reverse(model, rule_table, args, device):
    if args.task != "lm" or args.lm_direction != "reverse":
        raise ValueError("autoregressive reverse eval requires --task lm --lm-direction reverse")
    if args.mask_row_prefix:
        raise ValueError("autoregressive reverse eval is intended for unmasked causal LM runs")
    model.eval()
    correct_by_frame = torch.zeros(args.frames, device=device)
    count_by_frame = torch.zeros(args.frames, device=device)
    consistency_correct_by_frame = torch.zeros(args.frames, device=device)
    consistency_count_by_frame = torch.zeros(args.frames, device=device)
    exact_by_frame = torch.zeros(args.frames, device=device)
    exact_count_by_frame = torch.zeros(args.frames, device=device)
    processed = 0
    start = time.time()

    while processed < args.autoregressive_samples:
        batch_size = min(args.autoregressive_batch_size, args.autoregressive_samples - processed)
        initial = torch.randint(0, 2, (batch_size, args.width), device=device)
        history = eca_history(initial, args.frames, rule_table)
        if args.autoregressive_use_cache:
            reconstructed = reconstruct_reverse_autoregressive_cached(model, history, device, args.dtype)
        else:
            reconstructed = reconstruct_reverse_autoregressive(model, history, device, args.dtype)

        for frame in range(args.frames - 1):
            correct = reconstructed[:, frame] == history[:, frame]
            correct_by_frame[frame] += correct.sum()
            count_by_frame[frame] += correct.numel()

            produced_next = eca_step(reconstructed[:, frame], rule_table)
            consistency = produced_next == history[:, frame + 1]
            consistency_correct_by_frame[frame] += consistency.sum()
            consistency_count_by_frame[frame] += consistency.numel()
            exact_by_frame[frame] += consistency.all(dim=-1).sum()
            exact_count_by_frame[frame] += batch_size

        processed += batch_size
        if processed == args.autoregressive_samples or processed % (args.autoregressive_batch_size * 5) == 0:
            elapsed = time.time() - start
            print(
                f"autoregressive_processed={processed} "
                f"elapsed={elapsed:.1f}s "
                f"samples_per_sec={processed / elapsed if elapsed > 0 else 0.0:.1f}",
                flush=True,
            )

    correct_cpu = correct_by_frame.cpu()
    count_cpu = count_by_frame.cpu()
    consistency_correct_cpu = consistency_correct_by_frame.cpu()
    consistency_count_cpu = consistency_count_by_frame.cpu()
    exact_cpu = exact_by_frame.cpu()
    exact_count_cpu = exact_count_by_frame.cpu()
    frame_accuracies = [
        (correct_cpu[idx] / count_cpu[idx]).item() if count_cpu[idx].item() > 0 else float("nan")
        for idx in range(args.frames)
    ]
    frame_counts = [int(count_cpu[idx].item()) for idx in range(args.frames)]
    consistency_accuracies = [
        (consistency_correct_cpu[idx] / consistency_count_cpu[idx]).item()
        if consistency_count_cpu[idx].item() > 0
        else float("nan")
        for idx in range(args.frames)
    ]
    consistency_counts = [int(consistency_count_cpu[idx].item()) for idx in range(args.frames)]
    exact_rates = [
        (exact_cpu[idx] / exact_count_cpu[idx]).item()
        if exact_count_cpu[idx].item() > 0
        else float("nan")
        for idx in range(args.frames)
    ]
    exact_counts = [int(exact_count_cpu[idx].item()) for idx in range(args.frames)]

    if args.autoregressive_csv is not None:
        with open(args.autoregressive_csv, "w", encoding="utf-8") as f:
            f.write(
                "frame,cell_accuracy,cell_count,"
                "next_cell_consistency,next_cell_count,next_row_exact_rate,next_row_count\n"
            )
            for frame in range(args.frames):
                cell_text = "" if frame_counts[frame] == 0 else f"{frame_accuracies[frame]:.8f}"
                consistency_text = (
                    ""
                    if consistency_counts[frame] == 0
                    else f"{consistency_accuracies[frame]:.8f}"
                )
                exact_text = "" if exact_counts[frame] == 0 else f"{exact_rates[frame]:.8f}"
                f.write(
                    f"{frame},{cell_text},{frame_counts[frame]},"
                    f"{consistency_text},{consistency_counts[frame]},"
                    f"{exact_text},{exact_counts[frame]}\n"
                )
        print(f"wrote_autoregressive_csv={args.autoregressive_csv}", flush=True)

    if args.autoregressive_frame_accuracy_image is not None:
        render_frame_accuracy_png(
            args.autoregressive_frame_accuracy_image,
            frame_accuracies,
            frame_counts,
        )
        print(
            f"wrote_autoregressive_frame_accuracy_image={args.autoregressive_frame_accuracy_image}",
            flush=True,
        )
    if args.autoregressive_transition_image is not None:
        render_frame_accuracy_png(
            args.autoregressive_transition_image,
            consistency_accuracies,
            consistency_counts,
        )
        print(
            f"wrote_autoregressive_transition_image={args.autoregressive_transition_image}",
            flush=True,
        )
    if args.autoregressive_comparison_image is not None:
        render_frame_metric_comparison_png(
            args.autoregressive_comparison_image,
            [
                (frame_accuracies, frame_counts),
                (consistency_accuracies, consistency_counts),
            ],
        )
        print(
            f"wrote_autoregressive_comparison_image={args.autoregressive_comparison_image}",
            flush=True,
        )

    valid_correct = correct_cpu[count_cpu > 0].sum().item()
    valid_count = count_cpu[count_cpu > 0].sum().item()
    valid_consistency_correct = consistency_correct_cpu[consistency_count_cpu > 0].sum().item()
    valid_consistency_count = consistency_count_cpu[consistency_count_cpu > 0].sum().item()
    valid_exact = exact_cpu[exact_count_cpu > 0].sum().item()
    valid_exact_count = exact_count_cpu[exact_count_cpu > 0].sum().item()
    print(
        f"autoregressive_overall_cell={valid_correct / valid_count if valid_count else 0.0:.4f} "
        f"autoregressive_overall_next_cell_consistency="
        f"{valid_consistency_correct / valid_consistency_count if valid_consistency_count else 0.0:.4f} "
        f"autoregressive_overall_next_row_exact="
        f"{valid_exact / valid_exact_count if valid_exact_count else 0.0:.4f} "
        f"predicted_frames={int((count_cpu > 0).sum().item())} "
        f"samples={args.autoregressive_samples}",
        flush=True,
    )
    return frame_accuracies, frame_counts, consistency_accuracies, consistency_counts


@torch.no_grad()
def evaluate_one_row_autoregressive_reverse(model, rule_table, args, device):
    if args.task != "lm" or args.lm_direction != "reverse":
        raise ValueError("one-row autoregressive eval requires --task lm --lm-direction reverse")
    if args.mask_row_prefix:
        raise ValueError("one-row autoregressive eval is intended for unmasked causal LM runs")
    model.eval()
    correct_by_frame = torch.zeros(args.frames, device=device)
    count_by_frame = torch.zeros(args.frames, device=device)
    consistency_correct_by_frame = torch.zeros(args.frames, device=device)
    consistency_count_by_frame = torch.zeros(args.frames, device=device)
    exact_by_frame = torch.zeros(args.frames, device=device)
    exact_count_by_frame = torch.zeros(args.frames, device=device)
    processed = 0
    start = time.time()

    while processed < args.one_row_autoregressive_samples:
        batch_size = min(
            args.one_row_autoregressive_batch_size,
            args.one_row_autoregressive_samples - processed,
        )
        initial = torch.randint(0, 2, (batch_size, args.width), device=device)
        history = eca_history(initial, args.frames, rule_table)

        for frame in range(args.frames - 1):
            predicted_row = predict_natural_row_autoregressive(
                model,
                history,
                frame,
                device,
                args.dtype,
            )
            correct = predicted_row == history[:, frame]
            correct_by_frame[frame] += correct.sum()
            count_by_frame[frame] += correct.numel()

            produced_next = eca_step(predicted_row, rule_table)
            consistency = produced_next == history[:, frame + 1]
            consistency_correct_by_frame[frame] += consistency.sum()
            consistency_count_by_frame[frame] += consistency.numel()
            exact_by_frame[frame] += consistency.all(dim=-1).sum()
            exact_count_by_frame[frame] += batch_size

        processed += batch_size
        if (
            processed == args.one_row_autoregressive_samples
            or processed % (args.one_row_autoregressive_batch_size * 5) == 0
        ):
            elapsed = time.time() - start
            print(
                f"one_row_autoregressive_processed={processed} "
                f"elapsed={elapsed:.1f}s "
                f"samples_per_sec={processed / elapsed if elapsed > 0 else 0.0:.1f}",
                flush=True,
            )

    correct_cpu = correct_by_frame.cpu()
    count_cpu = count_by_frame.cpu()
    consistency_correct_cpu = consistency_correct_by_frame.cpu()
    consistency_count_cpu = consistency_count_by_frame.cpu()
    exact_cpu = exact_by_frame.cpu()
    exact_count_cpu = exact_count_by_frame.cpu()
    frame_accuracies = [
        (correct_cpu[idx] / count_cpu[idx]).item() if count_cpu[idx].item() > 0 else float("nan")
        for idx in range(args.frames)
    ]
    frame_counts = [int(count_cpu[idx].item()) for idx in range(args.frames)]
    consistency_accuracies = [
        (consistency_correct_cpu[idx] / consistency_count_cpu[idx]).item()
        if consistency_count_cpu[idx].item() > 0
        else float("nan")
        for idx in range(args.frames)
    ]
    consistency_counts = [int(consistency_count_cpu[idx].item()) for idx in range(args.frames)]
    exact_rates = [
        (exact_cpu[idx] / exact_count_cpu[idx]).item()
        if exact_count_cpu[idx].item() > 0
        else float("nan")
        for idx in range(args.frames)
    ]
    exact_counts = [int(exact_count_cpu[idx].item()) for idx in range(args.frames)]

    if args.one_row_autoregressive_csv is not None:
        with open(args.one_row_autoregressive_csv, "w", encoding="utf-8") as f:
            f.write(
                "frame,cell_accuracy,cell_count,"
                "next_cell_consistency,next_cell_count,next_row_exact_rate,next_row_count\n"
            )
            for frame in range(args.frames):
                cell_text = "" if frame_counts[frame] == 0 else f"{frame_accuracies[frame]:.8f}"
                consistency_text = (
                    ""
                    if consistency_counts[frame] == 0
                    else f"{consistency_accuracies[frame]:.8f}"
                )
                exact_text = "" if exact_counts[frame] == 0 else f"{exact_rates[frame]:.8f}"
                f.write(
                    f"{frame},{cell_text},{frame_counts[frame]},"
                    f"{consistency_text},{consistency_counts[frame]},"
                    f"{exact_text},{exact_counts[frame]}\n"
                )
        print(f"wrote_one_row_autoregressive_csv={args.one_row_autoregressive_csv}", flush=True)

    if args.one_row_autoregressive_frame_accuracy_image is not None:
        render_frame_accuracy_png(
            args.one_row_autoregressive_frame_accuracy_image,
            frame_accuracies,
            frame_counts,
        )
        print(
            "wrote_one_row_autoregressive_frame_accuracy_image="
            f"{args.one_row_autoregressive_frame_accuracy_image}",
            flush=True,
        )
    if args.one_row_autoregressive_transition_image is not None:
        render_frame_accuracy_png(
            args.one_row_autoregressive_transition_image,
            consistency_accuracies,
            consistency_counts,
        )
        print(
            "wrote_one_row_autoregressive_transition_image="
            f"{args.one_row_autoregressive_transition_image}",
            flush=True,
        )
    if args.one_row_autoregressive_comparison_image is not None:
        render_frame_metric_comparison_png(
            args.one_row_autoregressive_comparison_image,
            [
                (frame_accuracies, frame_counts),
                (consistency_accuracies, consistency_counts),
            ],
        )
        print(
            f"wrote_one_row_autoregressive_comparison_image={args.one_row_autoregressive_comparison_image}",
            flush=True,
        )

    valid_correct = correct_cpu[count_cpu > 0].sum().item()
    valid_count = count_cpu[count_cpu > 0].sum().item()
    valid_consistency_correct = consistency_correct_cpu[consistency_count_cpu > 0].sum().item()
    valid_consistency_count = consistency_count_cpu[consistency_count_cpu > 0].sum().item()
    valid_exact = exact_cpu[exact_count_cpu > 0].sum().item()
    valid_exact_count = exact_count_cpu[exact_count_cpu > 0].sum().item()
    print(
        f"one_row_autoregressive_overall_cell={valid_correct / valid_count if valid_count else 0.0:.4f} "
        f"one_row_autoregressive_overall_next_cell_consistency="
        f"{valid_consistency_correct / valid_consistency_count if valid_consistency_count else 0.0:.4f} "
        f"one_row_autoregressive_overall_next_row_exact="
        f"{valid_exact / valid_exact_count if valid_exact_count else 0.0:.4f} "
        f"predicted_frames={int((count_cpu > 0).sum().item())} "
        f"samples={args.one_row_autoregressive_samples}",
        flush=True,
    )
    return frame_accuracies, frame_counts, consistency_accuracies, consistency_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("lm", "transition"), default="lm")
    parser.add_argument("--lm-direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--mask-row-prefix", action="store_true")
    parser.add_argument("--rule", type=int, default=110)
    parser.add_argument("--compare-rules", default=None)
    parser.add_argument(
        "--position-encoding",
        choices=("grid", "sequential", "both"),
        default="grid",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--reconstruction-image", default=None)
    parser.add_argument("--reconstruction-samples", type=int, default=1)
    parser.add_argument("--attention-image", default=None)
    parser.add_argument("--attention-targets", default=None)
    parser.add_argument("--attention-all-heads", action="store_true")
    parser.add_argument("--attention-layer", type=int, default=None)
    parser.add_argument("--attention-head", type=int, default=None)
    parser.add_argument("--attention-average-initial-row", action="store_true")
    parser.add_argument("--attention-order", choices=("model", "natural"), default="model")
    parser.add_argument("--attention-overlay", action="store_true")
    parser.add_argument("--evolution-image", default=None)
    parser.add_argument("--checkpoint-in", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--best-checkpoint-out", default=None)
    parser.add_argument("--frame-accuracy-samples", type=int, default=0)
    parser.add_argument("--frame-accuracy-batch-size", type=int, default=256)
    parser.add_argument("--frame-accuracy-csv", default=None)
    parser.add_argument("--frame-accuracy-image", default=None)
    parser.add_argument("--transition-consistency-samples", type=int, default=0)
    parser.add_argument("--transition-consistency-batch-size", type=int, default=256)
    parser.add_argument("--transition-consistency-csv", default=None)
    parser.add_argument("--transition-consistency-image", default=None)
    parser.add_argument("--frame-metrics-comparison-image", default=None)
    parser.add_argument("--autoregressive-samples", type=int, default=0)
    parser.add_argument("--autoregressive-batch-size", type=int, default=64)
    parser.add_argument("--autoregressive-csv", default=None)
    parser.add_argument("--autoregressive-frame-accuracy-image", default=None)
    parser.add_argument("--autoregressive-transition-image", default=None)
    parser.add_argument("--autoregressive-comparison-image", default=None)
    parser.add_argument("--autoregressive-use-cache", action="store_true")
    parser.add_argument("--one-row-autoregressive-samples", type=int, default=0)
    parser.add_argument("--one-row-autoregressive-batch-size", type=int, default=64)
    parser.add_argument("--one-row-autoregressive-csv", default=None)
    parser.add_argument("--one-row-autoregressive-frame-accuracy-image", default=None)
    parser.add_argument("--one-row-autoregressive-transition-image", default=None)
    parser.add_argument("--one-row-autoregressive-comparison-image", default=None)
    parser.add_argument("--loss-curve-image", default=None)
    parser.add_argument("--image-scale", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.rank, args.local_rank, args.world_size = setup_distributed()
    args.distributed = args.world_size > 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.compare_rules is not None:
        if args.distributed:
            raise ValueError("--compare-rules is not supported with torchrun/DDP")
        curves = {}
        for rule in parse_rule_list(args.compare_rules):
            torch.manual_seed(args.seed)
            _, _, _, curve = train_experiment(args, rule, device)
            curves[rule] = curve
        if args.loss_curve_image is not None:
            render_loss_curve_png(args.loss_curve_image, curves)
            print(f"wrote_loss_curve_image={args.loss_curve_image}", flush=True)
        return

    torch.manual_seed(args.seed)
    rule_table, make_batch, model, curve = train_experiment(args, args.rule, device)
    if not is_main_process(args):
        if is_distributed():
            dist.destroy_process_group()
        return
    if args.checkpoint_out is not None:
        final_step = curve[-1][0] if curve else 0
        final_eval_loss = curve[-1][1] if curve else None
        save_training_checkpoint(
            args.checkpoint_out,
            model,
            args,
            args.rule,
            curve,
            final_step,
            final_eval_loss,
            None,
        )
    if args.loss_curve_image is not None:
        render_loss_curve_png(args.loss_curve_image, {args.rule: curve})
        print(f"wrote_loss_curve_image={args.loss_curve_image}", flush=True)

    if args.reconstruction_image is not None:
        if args.task != "lm" or args.lm_direction != "reverse":
            raise ValueError("reconstruction images currently require --task lm --lm-direction reverse")
        root, ext = os.path.splitext(args.reconstruction_image)
        if not ext:
            ext = ".png"
        for sample_idx in range(args.reconstruction_samples):
            initial = torch.randint(0, 2, (1, args.width), device=device)
            true_history = eca_history(initial, args.frames, rule_table)
            reconstructed_history = reconstruct_reverse(model, true_history, device)
            path = (
                args.reconstruction_image
                if args.reconstruction_samples == 1
                else f"{root}_{sample_idx + 1:02d}{ext}"
            )
            render_reconstruction_png(
                path,
                true_history[0],
                reconstructed_history[0],
                args.image_scale,
            )
            cell_acc = (true_history == reconstructed_history).float().mean().item()
            exact_frames = (
                (true_history == reconstructed_history).all(dim=-1).float().mean().item()
            )
            print(
                f"wrote_reconstruction_image={path} "
                f"sample_cell_acc={cell_acc:.3f} "
                f"sample_exact_frame_rate={exact_frames:.3f}",
                flush=True,
            )

    if args.attention_image is not None:
        write_attention_images(model, rule_table, args, device)

    frame_accuracy_result = None
    transition_consistency_result = None
    if args.frame_accuracy_samples > 0:
        frame_accuracy_result = evaluate_frame_accuracy(model, rule_table, args, device)
    if args.transition_consistency_samples > 0:
        transition_consistency_result = evaluate_transition_consistency(model, rule_table, args, device)
    if (
        args.frame_metrics_comparison_image is not None
        and frame_accuracy_result is not None
        and transition_consistency_result is not None
    ):
        render_frame_metric_comparison_png(
            args.frame_metrics_comparison_image,
            [
                frame_accuracy_result,
                (transition_consistency_result[0], transition_consistency_result[1]),
            ],
        )
        print(f"wrote_frame_metrics_comparison_image={args.frame_metrics_comparison_image}", flush=True)
    if args.autoregressive_samples > 0:
        evaluate_autoregressive_reverse(model, rule_table, args, device)
    if args.one_row_autoregressive_samples > 0:
        evaluate_one_row_autoregressive_reverse(model, rule_table, args, device)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
