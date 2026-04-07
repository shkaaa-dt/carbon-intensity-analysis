import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# LOAD CO2 DATA

co2 = pd.read_csv("data/co2.csv")

co2 = co2.rename(columns={
    "Entity": "country",
    "Year": "year",
    "Annual CO₂ emissions (per capita)": "co2_per_capita"
})

co2 = co2[["country", "year", "co2_per_capita"]]


# LOAD GDP DATA

gdp = pd.read_csv("data/gdp.csv")

gdp = gdp.rename(columns={
    "Entity": "country",
    "Year": "year",
    "GDP per capita": "gdp_per_capita"  # adjust if needed
})

gdp = gdp[["country", "year", "gdp_per_capita"]]


# MERGE CO2 + GDP

df = pd.merge(co2, gdp, on=["country", "year"], how="inner")
df = df.dropna()


# LOAD RENEWABLE ENERGY DATA

renew = pd.read_csv("data/renewables.csv")

renew = renew.rename(columns={
    "Entity": "country",
    "Year": "year",
    "Renewables": "renewable_share"  # adjust if needed
})

renew = renew[["country", "year", "renewable_share"]]

# Merge renewables
df = pd.merge(df, renew, on=["country", "year"], how="inner")
df = df.dropna()


# CREATE EMISSIONS INTENSITY

df["emissions_intensity"] = df["co2_per_capita"] / df["gdp_per_capita"]


# COMPARE COUNTRIES (INTENSITY OVER TIME)

countries = [
    "United Kingdom",
    "United States",
    "India",
    "China"
]

plt.figure(figsize=(12, 7))

for country in countries:
    df_country = df[df["country"] == country]
    plt.plot(df_country["year"],
             df_country["emissions_intensity"],
             label=country)

plt.title("Emissions Intensity Over Time")
plt.xlabel("Year")
plt.ylabel("CO₂ per $ of GDP")
plt.legend()
plt.grid(True)
plt.show()


# FORECAST INTENSITY TO 2035

country = "India"  # change this to test others

df_country = df[df["country"] == country].dropna()

X = df_country["year"].values.reshape(-1, 1)
y = df_country["emissions_intensity"].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(df_country["year"].max(), 2036).reshape(-1, 1)
predictions = model.predict(future_years)

plt.figure(figsize=(10, 6))
plt.plot(df_country["year"], y, label="Historical")
plt.plot(future_years, predictions, linestyle="--", label="Predicted")

plt.title(f"Emissions Intensity Forecast - {country}")
plt.xlabel("Year")
plt.ylabel("CO₂ per $ of GDP")
plt.legend()
plt.grid(True)
plt.show()


# GDP vs CO2 CORRELATION

gdp_co2_corr = df_country["gdp_per_capita"].corr(
    df_country["co2_per_capita"]
)

print(f"\nGDP vs CO₂ Correlation ({country}): {gdp_co2_corr:.3f}")


# RENEWABLES vs INTENSITY ANALYSIS

plt.figure(figsize=(10, 6))
plt.scatter(df_country["renewable_share"],
            df_country["emissions_intensity"])

plt.title(f"Renewables vs Emissions Intensity - {country}")
plt.xlabel("Renewable Energy Share (%)")
plt.ylabel("CO₂ per $ of GDP")
plt.grid(True)
plt.show()

renew_corr = df_country["renewable_share"].corr(
    df_country["emissions_intensity"]
)

print(f"Renewables vs Intensity Correlation ({country}): {renew_corr:.3f}")
