-- visualizer.lua
-- Terminal ASCII Visualization, Loss Curve Plotter & Interactive Drawing Pad

local Matrix = require("matrix")
local Visualizer = {}

local SHADES = { " ", ".", ":", "-", "=", "+", "*", "#", "@" }

local function pixel_to_ascii(val)
	local idx = math.floor(val * (#SHADES - 1) + 0.5) + 1
	if idx < 1 then
		idx = 1
	end
	if idx > #SHADES then
		idx = #SHADES
	end
	local ch = SHADES[idx]
	return ch .. ch
end

function Visualizer.render_prediction(flat_pixels, true_label, pred_label, probs_row)
	print("+----------------+")
	for r = 1, 8 do
		local line = "|"
		for c = 1, 8 do
			local val = flat_pixels[(r - 1) * 8 + c]
			line = line .. pixel_to_ascii(val)
		end
		line = line .. "|"

		local digit_idx = r - 1
		local p = probs_row[digit_idx + 1]
		local bar_len = math.floor(p * 20 + 0.5)
		local bar = string.rep("#", bar_len) .. string.rep(".", 20 - bar_len)
		line = line .. string.format("  [%d] %5.1f%% |%s|", digit_idx, p * 100, bar)
		print(line)
	end
	print("+----------------+")

	for digit_idx = 8, 9 do
		local p = probs_row[digit_idx + 1]
		local bar_len = math.floor(p * 20 + 0.5)
		local bar = string.rep("#", bar_len) .. string.rep(".", 20 - bar_len)
		print(string.format("                    [%d] %5.1f%% |%s|", digit_idx, p * 100, bar))
	end

	if true_label ~= nil then
		local status = (true_label == pred_label) and "CORRECT" or "WRONG"
		print(string.format("True: %d | Predicted: %d (%s)\n", true_label, pred_label, status))
	else
		print(string.format("Live Prediction: %d\n", pred_label))
	end
end

-- Plot an ASCII training vs validation loss curve
function Visualizer.print_loss_curve(train_losses, test_losses)
	local height = 10
	local width = #train_losses
	local max_loss = 0.01

	for i = 1, width do
		if train_losses[i] > max_loss then
			max_loss = train_losses[i]
		end
		if test_losses[i] > max_loss then
			max_loss = test_losses[i]
		end
	end

	print("\n=== LOSS CURVE ('T' = Train Loss, 'V' = Test Loss, '*' = Overlap) ===")
	for row = height, 1, -1 do
		local threshold = (row / height) * max_loss
		local line = string.format("%5.2f |", threshold)
		for col = 1, width do
			local t_row = math.max(1, math.ceil((train_losses[col] / max_loss) * height))
			local v_row = math.max(1, math.ceil((test_losses[col] / max_loss) * height))
			if t_row == row and v_row == row then
				line = line .. "*"
			elseif t_row == row then
				line = line .. "T"
			elseif v_row == row then
				line = line .. "V"
			else
				line = line .. " "
			end
		end
		print(line)
	end
	print("      +" .. string.rep("-", width))
	print(string.format("       Epoch 1%sEpoch %d\n", string.rep(" ", math.max(1, width - 12)), width))
end

function Visualizer.print_confusion_matrix(true_labels, pred_labels)
	local cm = {}
	for r = 0, 9 do
		cm[r] = {}
		for c = 0, 9 do
			cm[r][c] = 0
		end
	end

	for i = 1, #true_labels do
		local t = true_labels[i]
		local p = pred_labels[i]
		cm[t][p] = cm[t][p] + 1
	end

	print("\n=== CONFUSION MATRIX (Rows: True | Cols: Predicted) ===")
	local header = "     "
	for c = 0, 9 do
		header = header .. string.format("%4d", c)
	end
	print(header)
	print("    +" .. string.rep("----", 10))

	for r = 0, 9 do
		local row_str = string.format("  %d |", r)
		for c = 0, 9 do
			row_str = row_str .. string.format("%4d", cm[r][c])
		end
		print(row_str)
	end
	print("")
end

-- Interactive 8x8 ASCII Drawing Canvas with live network inference
function Visualizer.drawing_pad(net)
	local grid = {}
	for r = 1, 8 do
		grid[r] = { 0, 0, 0, 0, 0, 0, 0, 0 }
	end
	local cur_r, cur_c = 4, 4

	print("\n=== INTERACTIVE 8x8 DRAWING PAD ===")
	print("Controls (type one or more keys and press Enter):")
	print("  w/a/s/d : Move cursor     x : Paint bright (1.0)")
	print("  e       : Erase pixel     c : Clear entire canvas")
	print("  q       : Exit drawing pad\n")

	while true do
		-- Flatten grid & get live prediction
		local flat = {}
		local idx = 1
		for r = 1, 8 do
			for c = 1, 8 do
				flat[idx] = grid[r][c]
				idx = idx + 1
			end
		end

		local X_in = Matrix.from_table({ flat })
		local preds, _, probs = net:predict(X_in)

		-- Render grid with cursor brackets []
		print("+----------------+")
		for r = 1, 8 do
			local line = "|"
			for c = 1, 8 do
				if r == cur_r and c == cur_c then
					line = line .. (grid[r][c] > 0.5 and "[]" or "<>")
				else
					line = line .. pixel_to_ascii(grid[r][c])
				end
			end
			line = line .. "|"
			local d = r - 1
			local p = probs.data[d + 1]
			local bar_len = math.floor(p * 18 + 0.5)
			local bar = string.rep("#", bar_len) .. string.rep(".", 18 - bar_len)
			line = line .. string.format("  [%d] %5.1f%% |%s|", d, p * 100, bar)
			print(line)
		end
		print("+----------------+")
		for d = 8, 9 do
			local p = probs.data[d + 1]
			local bar_len = math.floor(p * 18 + 0.5)
			local bar = string.rep("#", bar_len) .. string.rep(".", 18 - bar_len)
			print(string.format("                    [%d] %5.1f%% |%s|", d, p * 100, bar))
		end
		print(string.format("Cursor: (%d,%d) | Predicted Digit: %d", cur_r, cur_c, preds))

		io.write("Move/Draw [w/a/s/d/x/e/c/q]: ")
		local cmd = io.read("*l")
		if not cmd then
			break
		end

		for i = 1, #cmd do
			local ch = cmd:sub(i, i):lower()
			if ch == "q" then
				return
			elseif ch == "w" then
				cur_r = math.max(1, cur_r - 1)
			elseif ch == "s" then
				cur_r = math.min(8, cur_r + 1)
			elseif ch == "a" then
				cur_c = math.max(1, cur_c - 1)
			elseif ch == "d" then
				cur_c = math.min(8, cur_c + 1)
			elseif ch == "x" then
				grid[cur_r][cur_c] = 1.0
				-- Subtle anti-aliased bleed to neighbors for realistic strokes
				local neighbors = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } }
				for _, off in ipairs(neighbors) do
					local nr, nc = cur_r + off, cur_c + off
					if nr >= 1 and nr <= 8 and nc >= 1 and nc <= 8 then
						grid[nr][nc] = math.min(1.0, grid[nr][nc] + 0.25)
					end
				end
			elseif ch == "e" then
				grid[cur_r][cur_c] = 0.0
			elseif ch == "c" then
				for r = 1, 8 do
					for c = 1, 8 do
						grid[r][c] = 0.0
					end
				end
			end
		end
	end
end

return Visualizer
