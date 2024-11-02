--[[
a Radial Basis Function allows for the smooth interpolation between many points of high-dimensional input -> output samples

interpolator = RBFNL.new(phi, pList, vList)
	takes phi, a function which takes two entries from pList and outputs a "distance" between them
	creates a new interpolator which takes arguments of the same form of the entries of pList
	returns a table of the form of the entries of vList

Example:
	local function phi(a, b)
		return (b - a).Magnitude^3
		-- warning! Do not return even powers of the distances!
	end

	local pList = {
		Vector3.new(0, 0, 0),
		Vector3.new(1, 0, 0),
		Vector3.new(0, 1, 0),
		Vector3.new(1, 1, 0),
	}

	local vList = {
		{1},
		{2},
		{3},
		{4},
	}

	local interpolator = RBFNL.new(phi, pList, vList)

	print(unpack(interpolator(Vector3.new(0, 0, 0))))
		--> 1
	print(unpack(interpolator(Vector3.new(1, 0, 0))))
		--> 2
	print(unpack(interpolator(Vector3.new(0.5, 0.5, 0))))
		--> 2.5
]]

local RBFNL = {}
RBFNL.__index = RBFNL

local getRBFWeightsNonLinear
local solveLinearSystem
local computeNonLinear

function RBFNL.new(
	phi: (any, any) -> number,
	pList: {any},
	vList: {{any}}
)
	local self = setmetatable({}, RBFNL)
	self._phi = phi
	self._pList = pList
	self._weights = getRBFWeightsNonLinear(phi, pList, vList)
	
	return self
end

function RBFNL:__call(p)
	debug.profilebegin("RBF.__call")
	local result = computeNonLinear(self._phi, self._pList, self._weights, p)
	debug.profileend()
	return result
end

RBFNL.phiDefaults = {}

function RBFNL.phiDefaults.r3Number(a: number, b: number)
	return math.abs(b - a)^3
end

function RBFNL.phiDefaults.r3Vector3(a: Vector3, b: Vector3)
	return (b - a).Magnitude^3
end

function RBFNL.phiDefaults.angle3Vector3(a: Vector3, b: Vector3)
	return a:Angle(b)^3
end

-- we do not include a linear estimator
-- this is the case when the embedded surface is not linear
function getRBFWeightsNonLinear(phi: (any, any) -> number, pList: {any}, vList: {{any}}): {{any}}?
	if not vList[1] then
		return nil
	end

	local n = #pList

	local M = {}
	for i = 1, 1 + n do
		M[i] = {}
	end

	for i = 1, n do
		for j = i, n do
			local phir = phi(pList[i], pList[j])
			M[i][j] = phir
			M[j][i] = phir
		end
	end

	for i = 1, n do
		M[n + 1][i] = 1
		M[i][n + 1] = 1
	end

	M[n + 1][n + 1] = 0

	local weights = {}

	for i = 1, n do
		weights[i] = table.clone(vList[i])
	end

	local d = #vList[1]
	local vNull = {}
	for i = 1, d do
		vNull[i] = 0*vList[1][i]
	end
	weights[n + 1] = vNull

	solveLinearSystem(M, weights)

	return weights
end

--[[
solves for x in M*x = y
converts M into identity matrix
converts y into x

	M = {
		{M00, M01, M02, ...n},
		{M10, M11, M12, ...n},
		{M20, M21, M22, ...n},
		...n
	}

	x = {
		{x00, x01, x02, ...m},
		{x00, x01, x02, ...m},
		{x00, x01, x02, ...m},
		...n
	}

performs gaussian elimination
attempts to find the largest element each step, and swaps rows and columns
(so it is not fast, but error should be pretty low)
]]
function solveLinearSystem(M: {{number}}, y: {{any}})
	--column swaps need to go here
	local swaps = {}
	local d = #y[1]
	local n = #M

	for dag = 1, n do
		local largestI
		local largestJ
		local largestValue = 0
		for i = dag, n do
			for j = 1, n do
				local value = math.abs(M[i][j])
				if value > largestValue then
					largestValue = value
					largestI = i
					largestJ = j
				end
			end
		end
		if not (largestI and largestJ) then
			--error("singular matrix")
			break
		end

		-- swap the rows
		M[dag], M[largestI] = M[largestI], M[dag]
		y[dag], y[largestI] = y[largestI], y[dag]
		-- swap the cols
		swaps[dag] = largestJ
		--swaps[dag], swaps[largestJ] = swaps[largestJ], swaps[dag]
		for row = 1, n do
			local R = M[row]
			R[dag], R[largestJ] = R[largestJ], R[dag]
		end

		-- unitize the row
		local R = M[dag]
		local v = R[dag] -- should be largestValue
		for col = dag, n do
			R[col] /= v
		end
		for col = 1, d do
			y[dag][col] /= v
		end

		-- subtract from other rows
		for row = 1, n do
			local S = M[row]
			local u = S[dag]
			if row == dag then
				continue
			end
			for col = dag, n do
				S[col] -= u*R[col]
			end
			for col = 1, d do
				y[row][col] -= u*y[dag][col]
			end
		end
	end

	for i = #swaps, 1, -1 do
		local swap = swaps[i]
		y[i], y[swap] = y[swap], y[i]
	end
end

function computeNonLinear(phi: (any, any) -> number, pList: {any}, weights: {{any}}, p: any)
	local n = #pList
	local l = #weights[1]
	local output = table.clone(weights[n + 1])

	for i = 1, n do
		local phir = phi(p, pList[i])
		for j = 1, l do
			output[j] += phir*weights[i][j]
		end
	end

	return output
end

return RBFNL
