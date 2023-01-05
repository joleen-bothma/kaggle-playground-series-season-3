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
	using StatsKit, MLJ, Gadfly, Random, Lasso
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
- Show the correlation between each variable in the dataset. This will reveal two things to us:
  1. Are any of the feature variables highly correlated with the target variable? This will inform the strength of a statistical model.
  2. Are any of the feature variables highly correlated with each other? This will tell us if multicollinearity is present in the data. If so, this is a violation of a key assumption of linear regression and we must consider regularization techniques to deal with this.
- Plot a QQ plot of the target variable to check if it is normally distributed.
- Print the Skewness (goal: 0) and Kurtosis (goal: 3) statistics of the target variable as an additional check for normality. If the target variable has a high degree of skewness and kurtosis, a log transformation may need to be considered to improve model performance.

Note that we are checking these assumptions to inform a few of our decisions later on in the modeling process, such as whether to use regularization or if a log transform is necessary of one or more variables. However, for very large datasets, some degree of violation of these assumptions will not hurt a statistical model too much, so don't obsess over achieving perfection here.

After a preliminary linear regression model is fitted to the data, we can perform these checks for if any assumptions are violated:
- Residual plots:
- Variance inflation factor (VIF) of each explanatory variable on the target. If the VIF score is over 5, a high amount of multicollinearity is present in the data.
"""

# ╔═╡ 61aa5283-1784-4699-aa88-e2d131d981b6
describe(raw_df, :all)

# ╔═╡ 8bc24e55-3d55-4c86-bf10-c1c6fca9227f
md"""
### Feature Engineering

- Number of houses in a block - ie. block density
- Number of unused bedrooms - ie. house is too big for number of occupants

- To come later: 
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

# ╔═╡ 7e62e0f5-c395-465b-b363-98df3ce60fe2
md"""
## Modeling Approach

Since the target variable is continuous, there are a few approaches we can take to make predictions. While it is fine to use just one model, we will be taking a more comprehensive approach, using an ensemble of multiple techniques and models to make a final prediction.

- Linear regression
- Ridge / Lasso regularized regression
- K-Nearest Neighbors regression
- Support Vector Machine (SVM)
- Decision trees (CART algorithm)
- XGBoost

Models will be combined into an ensemble using Stacking, also known as Stacked Generalization.

"""

# ╔═╡ 0c87863b-568d-434e-8836-0e9f903d2d0d
md"""
Q: Is it possible to use K-Means Clustering on the variables Longitude, Latitude, and MedHouseVal in order to identify location clusters as an alternative feature in the model?

A [discussion post on Kaggle](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376078) shows that coastal + big cities are highly correlated with high housing values.
"""

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
# ╠═a40fa645-f6c1-4982-867c-4ff7bad8fac5
# ╠═62c60ce5-efed-4589-bd2a-bb14d4b124cd
# ╠═65b379be-24ea-4270-b7f5-be16e4752f67
# ╠═4ee4049a-9497-4154-8f63-20fb75a1d99c
# ╠═55a403cb-f2ce-4519-8a6d-6119121afe17
# ╠═37ef437f-c6ef-4377-9fa2-7c923a2fbd8a
# ╠═ecac2d41-ed8e-4454-8041-b1693b2a14cb
# ╠═b5b43e1e-0099-4386-a04b-683298266ad4
# ╠═db7fdac5-882b-4de0-be9f-ee75c73da32e
# ╠═a681d7aa-604e-4968-8ebc-b63e518d6807
# ╠═00928d3d-ae28-4645-8969-751f913dc146
# ╠═8ff9d4ef-9cc9-4add-b2fc-d5225e43ecfd
# ╠═7e62e0f5-c395-465b-b363-98df3ce60fe2
# ╠═0c87863b-568d-434e-8836-0e9f903d2d0d
