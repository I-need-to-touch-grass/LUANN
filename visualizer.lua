-- visualizer.lua
-- Terminal ASCII Visualization for Digits, Probabilities & Confusion Matrix

local Visualizer = {}

local SHADES = { " ", ".", ":", "-", "=", "+", "*", "#", "@" }

-- Map a float [0.0, 1.0] to a 2-character ASCII block
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

-- Print an 8x8 digit grid with border and prediction confidence bars
function Visualizer.render_prediction(flat_pixels, true_label, pred_label, probs_row)
	print("+----------------+")
	for r = 1, 8 do
		local line = "|"
		for c = 1, 8 do
			local val = flat_pixels[(r - 1) * 8 + c]
			line = line .. pixel_to_ascii(val)
		end
		line = line .. "|"

		-- Print probability bar for digits 0-7 beside the image rows
		local digit_idx = r - 1
		local p = probs_row[digit_idx + 1]
		local bar_len = math.floor(p * 20 + 0.5)
		local bar = string.rep("#", bar_len) .. string.rep(".", 20 - bar_len)
		line = line .. string.format("  [%d] %5.1f%% |%s|", digit_idx, p * 100, bar)
		print(line)
	end
	print("+----------------+")

	-- Remaining digits 8 and 9
	for digit_idx = 8, 9 do
		local p = probs_row[digit_idx + 1]
		local bar_len = math.floor(p * 20 + 0.5)
		local bar = string.rep("#", bar_len) .. string.rep(".", 20 - bar_len)
		print(string.format("                    [%d] %5.1f%% |%s|", digit_idx, p * 100, bar))
	end

	local status = (true_label == pred_label) and "CORRECT" or "WRONG"
	print(string.format("True: %d | Predicted: %d (%s)\n", true_label, pred_label, status))
end

-- Build and print a 10x10 Confusion Matrix
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

return Visualizer
