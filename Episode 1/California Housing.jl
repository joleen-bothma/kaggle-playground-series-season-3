### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ 3a742e1e-0cd0-4539-8dd6-774ba5820fd2
# ╠═╡ show_logs = false
begin
	import Pkg

	# Activate the environment at the current working directory
	Pkg.activate(Base.current_project())

	# Instantiate to prevent missing dependencies
	Pkg.instantiate

	# Specify the packages you will be using in this project
	using StatsKit, MLJ, Gadfly, Random, Lasso, MLBase
	import Plots
end

# ╔═╡ 3dcda415-95e0-4bd7-80ee-16715668755a
md"# California Housing - Predicting Median House Value"

# ╔═╡ 60e45daf-baf7-4b58-bd02-f9985ffdd223
md"""
## Project Overview
This project is based on [Season 3, Episode 1](https://www.kaggle.com/competitions/playground-series-s3e1) of the Playground Series on Kaggle.
"""

# ╔═╡ 4c65c177-9334-4681-b187-a31c0b44f603
md"## Packages"

# ╔═╡ 5cbb1670-55b4-4217-a65c-adea517d28ef
md"""
Other settings:
"""

# ╔═╡ 0cccd6de-bef4-4d20-b0d4-4ef7c8a4e1d7
# Set the random seed for this notebook (for reproducibility)
Random.seed!(42)

# ╔═╡ 4fba0b72-b97e-46d3-be74-0a6f0ecdbea9
# Set the default plot size for all Gadfly plots
set_default_plot_size(18cm, 14cm)

# ╔═╡ d9d5b5a8-d4f8-47a2-b0ca-d92f6de1c862
md"## Import data"

# ╔═╡ c8f4a905-48cb-4e24-8087-fdb227d833ba
raw_df = CSV.read("data/train.csv", DataFrame)

# ╔═╡ a0474ad8-2e22-4d7d-876a-24c942fbf38e
md"## Viewing and Exploring the Data"

# ╔═╡ 1a91904e-bbdc-4d0a-bdaf-8d350965b439
md"""
This preliminary peek at the data does not expose any obvious issues that need to be addressed.

However, in our initial exploration of the data, we must determine if there are any outliers or 'funnies' that shouldn't be there.

These are our tasks:
- Plot the variables in a Box Plot to determine outliers.
- Plus, use Tukey's outer fences method to pinpoint outlier values - using quartiles and interquartile range (IQR)
- Show the correlation between each variable in the dataset. This will reveal two things to us:
  1. Are any of the feature variables highly correlated with the target variable? This will inform the strength of a statistical model.
  2. Are any of the feature variables highly correlated with each other? This will tell us if multicollinearity is present in the data. If so, this is a violation of a key assumption of linear regression and we must consider regularization techniques to deal with this.

"""

# ╔═╡ 61aa5283-1784-4699-aa88-e2d131d981b6
describe(raw_df, :all)

# ╔═╡ 8bc24e55-3d55-4c86-bf10-c1c6fca9227f
md"""
### Feature Engineering

- Number of houses in a block - ie. block density
- Number of unused bedrooms - ie. house is too big for number of occupants

- To come later maybe: 
  - Location classification of block (coastal, big city, other?) using longitude and latitude (and maybe also target variable to classify further?)
  - Classification of house size - how big is a house? Create a classification using AveRooms and AveBedrms
"""

# ╔═╡ 8996880f-5d35-4ce2-b76e-06c4f979bc53
# Make a copy of the dataframe to begin our feature engineering
mod_df = copy(raw_df)

# ╔═╡ b23e83f2-97bf-489f-ae3d-0cad3293481f
# Number of houses in a block (ie. block density)
# Hyp: Dense blocks = smaller houses = lower house value?
# Or: Dense blocks = tall apartment buildings = close to city = higher house value?
mod_df.BlockDensity = mod_df.Population ./ mod_df.AveOccup

# ╔═╡ 2023d12c-79b9-4f86-9554-f3f05a332bfa
# Number of unused bedrooms
mod_df.AveUnusedBedrms = mod_df.AveBedrms .- mod_df.AveOccup

# ╔═╡ 02ef74f6-c8a4-4a03-a544-01878fc23e77
# If unused bedrooms is negative then there are no unused bedrooms
mod_df.AveUnusedBedrms[mod_df.AveUnusedBedrms .< 0] .= 0

# ╔═╡ 0a6d5830-b016-48c6-93f6-4b377217f78f
# View modified dataframe
sort(mod_df, order(:AveBedrms, rev = true))

# ╔═╡ 8647127e-4bba-4687-b5e1-3b19c50a3b28
size(mod_df[!, "AveBedrms"][mod_df[!, "AveBedrms"] .< 1, :])[1]

# ╔═╡ a40fa645-f6c1-4982-867c-4ff7bad8fac5
md"""
### Outliers
"""

# ╔═╡ 62c60ce5-efed-4589-bd2a-bb14d4b124cd
# MedHouseVal
title(
	hstack(
		plot(raw_df,
			y = "MedHouseVal",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "MedHouseVal",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"(Target) Median House Value of Block Group"
)

# ╔═╡ 4ee4049a-9497-4154-8f63-20fb75a1d99c
# HouseAge
title(
	hstack(
		plot(raw_df,
			y = "HouseAge",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "HouseAge",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Median House Age of Block Group"
)

# ╔═╡ 55a403cb-f2ce-4519-8a6d-6119121afe17
# AveRooms
title(
	hstack(
		plot(raw_df,
			y = "AveRooms",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "AveRooms",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Avg Rooms of Block Group"
)

# ╔═╡ 37ef437f-c6ef-4377-9fa2-7c923a2fbd8a
# AveBedrms
title(
	hstack(
		plot(raw_df,
			y = "AveBedrms",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "AveBedrms",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Avg Bedrooms of Block Group"
)

# ╔═╡ ecac2d41-ed8e-4454-8041-b1693b2a14cb
# Population
title(
	hstack(
		plot(raw_df,
			y = "Population",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "Population",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Population of Block Group"
)

# ╔═╡ b5b43e1e-0099-4386-a04b-683298266ad4
# AveOccup
title(
	hstack(
		plot(raw_df,
			y = "AveOccup",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "AveOccup",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Avg Occupants of Block Group"
)

# ╔═╡ db7fdac5-882b-4de0-be9f-ee75c73da32e
# BlockDensity
title(
	hstack(
		plot(mod_df,
			y = "BlockDensity",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(mod_df,
			x = "BlockDensity",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Density of Block Group"
)

# ╔═╡ a681d7aa-604e-4968-8ebc-b63e518d6807
# AveUnusedBedrms
title(
	hstack(
		plot(mod_df,
			y = "AveUnusedBedrms",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(mod_df,
			x = "AveUnusedBedrms",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Avg Unused Bedrooms in Block Group"
)

# ╔═╡ 00928d3d-ae28-4645-8969-751f913dc146
md"""
### Correlations
"""

# ╔═╡ 8ff9d4ef-9cc9-4add-b2fc-d5225e43ecfd
begin
	cols = names(mod_df)
	corr_matrix = cor(Matrix(mod_df))
	
	# PLOT
	(n,m) = size(corr_matrix)
	Plots.heatmap(corr_matrix, xticks=(1:m,cols), xrot=90, yticks=(1:m,cols), yflip=true, c = :bluesreds)
	Plots.annotate!([(j, i, (round(corr_matrix[i,j],digits=1), 8,"Computer Modern",:white)) for i in 1:n for j in 1:m])
end

# ╔═╡ 65b379be-24ea-4270-b7f5-be16e4752f67
# MedInc
title(
	hstack(
		plot(raw_df,
			y = "MedInc",
			Geom.boxplot,
			Guide.title("Boxplot"),
			Theme(point_size = 10px,
				  default_color = "palegreen3")),
		
		plot(raw_df,
			x = "MedInc",
			Geom.histogram,
			Guide.title("Histogram"),
			Theme(default_color = "palegreen3"))
	),
	"Median Income of Block Group"
)

# ╔═╡ ba2784a4-c7ec-48ea-a2df-3b0fe39d83ba
md"""
### Data Preparation
"""

# ╔═╡ 6eff9140-45f2-466d-9e91-985d846d2d73
# Finding outliers past the outer fences
# (Q1 - (3*IQR); Q3 + (3*IQR))

function outer_fences(df, X) 
	print("\n $(X)")
	# Quantiles
	quantiles = quantile!(df[!, X], [0.25, 0.75])
	q1 = quantiles[1]
	q3 = quantiles[2]

	# Interquartile range
	IQR = q3-q1

	# Fences
	fence_1 = q1 - (3*IQR)
	fence_2 = q3 + (3*IQR)

	print("\n Fence cutoffs: ($(round(fence_1, digits=2)), $(round(fence_2,digits=2)))")

	# Count the number of samples that are outside the fences
	outside_samples = df[!, X][(df[!, X] .< fence_1) .| (df[!, X] .> fence_2), :]
	count_outside_samples = size(outside_samples)[1]
	print("\n Number of outside samples: $(count_outside_samples)")	

	return X, fence_1, fence_2
end

# ╔═╡ dfbfc611-578b-4b90-b8e2-c8d52b068a6b
# Get outside samples
begin
	outerfences_df = DataFrame("Column" => [], "Fence1" => [], "Fence2" => [])

	for column in names(mod_df)
		outerfences = outer_fences(mod_df, column)
		push!(outerfences_df, [outerfences[1], outerfences[2], outerfences[3]])
	end
end

# ╔═╡ ad1efa11-0a77-4059-94d1-2aa6e1e81e85
outerfences_df

# ╔═╡ 0fd5c33b-15c7-49db-83e3-1e3a88265a89
mod_df

# ╔═╡ c611554b-22bb-4acf-8ec5-4bf8e58b3d15
# Remove outer fence outliers
# Population
yay = mod_df[
	(
		mod_df[!, "Population"] .> 
			outerfences_df[outerfences_df.Column .== "Population", :].Fence1
		
	).&&
	(
		mod_df[!, "Population"] .< 
			outerfences_df[outerfences_df.Column .== "Population", :].Fence2
		
	)
	,:]

# ╔═╡ ae78685f-feef-408a-aacb-8481f1e79f26
# Remove outer fence outliers
# AveBedrms
yay2 = yay[
	(
		yay[!, "AveBedrms"] .> 
			outerfences_df[outerfences_df.Column .== "AveBedrms", :].Fence1
		
	).&&
	(
		yay[!, "AveBedrms"] .< 
			outerfences_df[outerfences_df.Column .== "AveBedrms", :].Fence2
		
	)
	,:]

# ╔═╡ dd8d92fb-b8db-4a22-a5ca-3f29537afb80
# Make a copy of the dataframe to begin our cleaning
clean_df = copy(yay2)

# ╔═╡ 2dacbb7b-7185-4d53-bd56-f41d8b3a4f3c
# Split into X and Y dataframes
Y, X = unpack(clean_df, ==(:MedHouseVal), !=(:id))

# ╔═╡ 4d567824-a7ed-494d-92ec-b07ad878a85b
# Split into training and test sets
(train, test) = partition(clean_df, 0.7, shuffle=true)

# ╔═╡ ae39a820-dc89-4cbb-a366-11210572b269
size(train)

# ╔═╡ 015d8f81-b0f0-4aa9-9eba-2ab16790a95e
size(test)

# ╔═╡ 7e62e0f5-c395-465b-b363-98df3ce60fe2
md"""
## Modeling

Since the target variable is continuous, there are a few approaches we can take to make predictions. For this project, I just use a simple linear regression to get familiar with Julia.

"""

# ╔═╡ 0c87863b-568d-434e-8836-0e9f903d2d0d
md"""
Q: Is it possible to use K-Means Clustering on the variables Longitude, Latitude, and MedHouseVal in order to identify location clusters as an alternative feature in the model?

A [discussion post on Kaggle](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376078) shows that coastal + big cities are highly correlated with high housing values.
"""

# ╔═╡ ef806872-ccae-4106-98ad-3621904ea508
names(train)

# ╔═╡ 2b8514a0-68fe-4646-960b-61e6ec292e66
linearRegressor = lm(@formula(MedHouseVal ~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude + BlockDensity), train)

# ╔═╡ 0f0f99ee-511a-4d9a-b271-bcb98e2a1da3
# R Square value of the model
r2(linearRegressor)

# ╔═╡ 2e894b97-0f6d-4079-a2dc-0d017a031283
# RMSE function defination
function rmse(performance_df)
    rmse = sqrt(mean(performance_df.error.*performance_df.error))
    return rmse
end

# ╔═╡ 2c139d77-cbfe-4de5-bd00-66a7bdf17937
begin
	# Prediction
	ypredicted_test = GLM.predict(linearRegressor, test)
	ypredicted_train = GLM.predict(linearRegressor, train)
	
	# Test Performance DataFrame
	performance_testdf = DataFrame(y_actual = test[!,:MedHouseVal], y_predicted = ypredicted_test)
	performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
	performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error
	
	# Train Performance DataFrame
	performance_traindf = DataFrame(y_actual = train[!,:MedHouseVal], y_predicted = ypredicted_train)
	performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
	performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error
end

# ╔═╡ f0d5f34d-3cff-4ee5-ba6c-dcf0e914ce9f
begin
	# Test Error
	println("Root mean square test error: ",rmse(performance_testdf), "\n")
	println("Mean square test error: ",mean(performance_testdf.error_sq), "\n")
end

# ╔═╡ e4362cc0-7bfb-44ed-84a5-9828aded7112
begin
	# Train Error
	println("Root mean square test error: ",rmse(performance_traindf), "\n")
	println("Mean square test error: ",mean(performance_traindf.error_sq), "\n")
end

# ╔═╡ 234a9d37-982b-4d08-b33f-ecfc0d769267
# Scatter plot of actual vs predicted values on test dataset
test_plot = plot(performance_testdf, 
		x = performance_testdf[!,:y_actual], 
		y = performance_testdf[!,:y_predicted], 
		Geom.point)


# ╔═╡ 7282cef3-0995-42fe-97de-e50377e63b4e
# Cross Validation function defination
function cross_validation(train,k, fm = @formula(MedHouseVal ~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude + BlockDensity))
    a = collect(Kfold(size(train)[1], k))
    for i in 1:k
        row = a[i]
        temp_train = train[row,:]
        temp_test = train[setdiff(1:end, row),:]
        linearRegressor = lm(fm, temp_train)
        performance_testdf = DataFrame(y_actual = temp_test[!,:MedHouseVal], y_predicted = GLM.predict(linearRegressor, temp_test))
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]

		println("Root mean square test error for set $i is: ",rmse(performance_testdf), "\n")
    end
end

# ╔═╡ cc9baba5-b3cd-451f-a551-77fd89a59e1a
cross_validation(train,10)

# ╔═╡ 908bbbb7-44eb-4b2f-b9c4-d2bc87154f3d
raw_df_test = CSV.read("data/test.csv", DataFrame)

# ╔═╡ 6059ecbc-339f-46f7-b9c1-952b8ba876d7
raw_df_test.BlockDensity = raw_df_test.Population ./ raw_df_test.AveOccup

# ╔═╡ 31abc289-e877-460e-ac85-46d58cb52c65
size(raw_df_test)

# ╔═╡ 416ce5b7-3ded-419c-a012-80a9240ab6d0
# Prediction
ypredicted_all = GLM.predict(linearRegressor, raw_df_test)

# ╔═╡ d8eceeb5-c497-4845-829b-c99dc6ca5612
base_submission = DataFrame(id = raw_df_test[!,:id], MedHouseVal = ypredicted_all)

# ╔═╡ 7b172268-0320-44dc-8a1a-e10320ee9c19
# Export base submission
CSV.write("data/base_submission.csv", base_submission)

# ╔═╡ Cell order:
# ╟─3dcda415-95e0-4bd7-80ee-16715668755a
# ╠═60e45daf-baf7-4b58-bd02-f9985ffdd223
# ╟─4c65c177-9334-4681-b187-a31c0b44f603
# ╠═3a742e1e-0cd0-4539-8dd6-774ba5820fd2
# ╠═5cbb1670-55b4-4217-a65c-adea517d28ef
# ╠═0cccd6de-bef4-4d20-b0d4-4ef7c8a4e1d7
# ╠═4fba0b72-b97e-46d3-be74-0a6f0ecdbea9
# ╟─d9d5b5a8-d4f8-47a2-b0ca-d92f6de1c862
# ╠═c8f4a905-48cb-4e24-8087-fdb227d833ba
# ╠═a0474ad8-2e22-4d7d-876a-24c942fbf38e
# ╠═1a91904e-bbdc-4d0a-bdaf-8d350965b439
# ╠═61aa5283-1784-4699-aa88-e2d131d981b6
# ╠═8bc24e55-3d55-4c86-bf10-c1c6fca9227f
# ╠═8996880f-5d35-4ce2-b76e-06c4f979bc53
# ╠═b23e83f2-97bf-489f-ae3d-0cad3293481f
# ╠═2023d12c-79b9-4f86-9554-f3f05a332bfa
# ╠═02ef74f6-c8a4-4a03-a544-01878fc23e77
# ╠═0a6d5830-b016-48c6-93f6-4b377217f78f
# ╠═8647127e-4bba-4687-b5e1-3b19c50a3b28
# ╠═a40fa645-f6c1-4982-867c-4ff7bad8fac5
# ╠═62c60ce5-efed-4589-bd2a-bb14d4b124cd
# ╠═4ee4049a-9497-4154-8f63-20fb75a1d99c
# ╠═55a403cb-f2ce-4519-8a6d-6119121afe17
# ╠═37ef437f-c6ef-4377-9fa2-7c923a2fbd8a
# ╠═ecac2d41-ed8e-4454-8041-b1693b2a14cb
# ╠═b5b43e1e-0099-4386-a04b-683298266ad4
# ╠═db7fdac5-882b-4de0-be9f-ee75c73da32e
# ╠═a681d7aa-604e-4968-8ebc-b63e518d6807
# ╠═00928d3d-ae28-4645-8969-751f913dc146
# ╠═8ff9d4ef-9cc9-4add-b2fc-d5225e43ecfd
# ╠═65b379be-24ea-4270-b7f5-be16e4752f67
# ╠═ba2784a4-c7ec-48ea-a2df-3b0fe39d83ba
# ╠═6eff9140-45f2-466d-9e91-985d846d2d73
# ╠═dfbfc611-578b-4b90-b8e2-c8d52b068a6b
# ╠═ad1efa11-0a77-4059-94d1-2aa6e1e81e85
# ╠═0fd5c33b-15c7-49db-83e3-1e3a88265a89
# ╠═c611554b-22bb-4acf-8ec5-4bf8e58b3d15
# ╠═ae78685f-feef-408a-aacb-8481f1e79f26
# ╠═dd8d92fb-b8db-4a22-a5ca-3f29537afb80
# ╠═2dacbb7b-7185-4d53-bd56-f41d8b3a4f3c
# ╠═4d567824-a7ed-494d-92ec-b07ad878a85b
# ╠═ae39a820-dc89-4cbb-a366-11210572b269
# ╠═015d8f81-b0f0-4aa9-9eba-2ab16790a95e
# ╠═7e62e0f5-c395-465b-b363-98df3ce60fe2
# ╠═0c87863b-568d-434e-8836-0e9f903d2d0d
# ╠═ef806872-ccae-4106-98ad-3621904ea508
# ╠═2b8514a0-68fe-4646-960b-61e6ec292e66
# ╠═0f0f99ee-511a-4d9a-b271-bcb98e2a1da3
# ╠═2e894b97-0f6d-4079-a2dc-0d017a031283
# ╠═2c139d77-cbfe-4de5-bd00-66a7bdf17937
# ╠═f0d5f34d-3cff-4ee5-ba6c-dcf0e914ce9f
# ╠═e4362cc0-7bfb-44ed-84a5-9828aded7112
# ╠═234a9d37-982b-4d08-b33f-ecfc0d769267
# ╠═7282cef3-0995-42fe-97de-e50377e63b4e
# ╠═cc9baba5-b3cd-451f-a551-77fd89a59e1a
# ╠═908bbbb7-44eb-4b2f-b9c4-d2bc87154f3d
# ╠═6059ecbc-339f-46f7-b9c1-952b8ba876d7
# ╠═31abc289-e877-460e-ac85-46d58cb52c65
# ╠═416ce5b7-3ded-419c-a012-80a9240ab6d0
# ╠═d8eceeb5-c497-4845-829b-c99dc6ca5612
# ╠═7b172268-0320-44dc-8a1a-e10320ee9c19
