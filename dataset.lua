-- dataset.lua
-- 8x8 Handwritten Digit Generator with Rotation, Shift, Occlusion & Noise

local Matrix = require("matrix")
local Dataset = {}

local PROTOTYPES = {
	[0] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[1] = {
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 1, 0, 0, 0 },
		{ 0, 1, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
	},
	[2] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
	},
	[3] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[4] = {
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 1, 0, 1, 1, 0, 0 },
		{ 0, 1, 0, 0, 1, 1, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
	},
	[5] = {
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[6] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[7] = {
		{ 0, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0 },
	},
	[8] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
	},
	[9] = {
		{ 0, 0, 1, 1, 1, 1, 0, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 1, 1, 1, 0, 0, 0 },
	},
}

local function clamp(val, min_v, max_v)
	if val < min_v then
		return min_v
	end
	if val > max_v then
		return max_v
	end
	return val
end

-- Bilinear interpolation helper for smooth rotation
local function sample_bilinear(grid, r, c)
	local r0 = math.floor(r)
	local c0 = math.floor(c)
	local dr = r - r0
	local dc = c - c0

	local function get_px(rr, cc)
		if rr < 1 or rr > 8 or cc < 1 or cc > 8 then
			return 0.0
		end
		return grid[rr][cc]
	end

	local v00 = get_px(r0, c0)
	local v01 = get_px(r0, c0 + 1)
	local v10 = get_px(r0 + 1, c0)
	local v11 = get_px(r0 + 1, c0 + 1)

	return (1 - dr) * ((1 - dc) * v00 + dc * v01) + dr * ((1 - dc) * v10 + dc * v11)
end

-- Apply random rotation (-15 to +15 deg), translation (-1 to +1 px), occlusion & noise
function Dataset.augment_grid(grid, noise_level)
	local angle_rad = math.rad((math.random() * 2.0 - 1.0) * 15.0)
	local cos_a = math.cos(angle_rad)
	local sin_a = math.sin(angle_rad)

	local shift_r = (math.random() * 2.0 - 1.0) * 0.9
	local shift_c = (math.random() * 2.0 - 1.0) * 0.9
	local intensity_scale = 0.85 + math.random() * 0.3

	-- Optional 2x2 random cutout patch (30% chance)
	local use_cutout = math.random() < 0.30
	local cut_r = math.random(1, 7)
	local cut_c = math.random(1, 7)

	local cx, cy = 4.5, 4.5
	local out = {}

	for r = 1, 8 do
		out[r] = {}
		for c = 1, 8 do
			local dr = (r - shift_r) - cy
			local dc = (c - shift_c) - cx
			local src_r = cy + (dr * cos_a - dc * sin_a)
			local src_c = cx + (dr * sin_a + dc * cos_a)

			local base = sample_bilinear(grid, src_r, src_c) * intensity_scale

			-- Apply cutout occlusion
			if use_cutout and (r == cut_r or r == cut_r + 1) and (c == cut_c or c == cut_c + 1) then
				base = base * 0.15
			end

			-- Random stroke dropout or bleed
			if base > 0.5 and math.random() < 0.06 then
				base = base * 0.4
			elseif base == 0 and math.random() < 0.04 then
				base = 0.2
			end

			local noise = (math.random() * 2.0 - 1.0) * (noise_level or 0.15)
			out[r][c] = clamp(base + noise, 0.0, 1.0)
		end
	end

	return out
end

function Dataset.flatten(grid)
	local flat = {}
	local idx = 1
	for r = 1, 8 do
		for c = 1, 8 do
			flat[idx] = grid[r][c]
			idx = idx + 1
		end
	end
	return flat
end

function Dataset.generate(num_samples, noise_level)
	local samples = {}
	for i = 1, num_samples do
		local digit = (i - 1) % 10
		local aug = Dataset.augment_grid(PROTOTYPES[digit], noise_level)
		local flat = Dataset.flatten(aug)
		local one_hot = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		one_hot[digit + 1] = 1.0
		samples[i] = {
			x = flat,
			y = one_hot,
			label = digit,
		}
	end

	for i = #samples, 2, -1 do
		local j = math.random(i)
		samples[i], samples[j] = samples[j], samples[i]
	end

	return samples
end

function Dataset.to_matrices(samples, start_idx, end_idx)
	local count = end_idx - start_idx + 1
	local X = Matrix.new(count, 64)
	local Y = Matrix.new(count, 10)
	local labels = {}

	for i = 1, count do
		local s = samples[start_idx + i - 1]
		X.data[i] = s.x
		Y.data[i] = s.y
		labels[i] = s.label
	end

	return X, Y, labels
end

function Dataset.get_prototype(digit)
	return PROTOTYPES[digit]
end

return Dataset
