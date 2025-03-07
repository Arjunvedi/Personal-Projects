install.packages("gapminder")
library(gapminder)
library(dplyr)
library(ggplot2)

data("gapminder")

gapminder <- gapminder %>%
  mutate(total_gdp = gdpPercap * pop)

gapminder_cleaned <- gapminder %>%
  filter(year >= 2000)

# Question 1
summary_stats <- gapminder_cleaned %>%
  group_by(continent) %>%
  summarise(
    avg_lifeExp = mean(lifeExp, na.rm = TRUE),
    avg_gdpPercap = mean(gdpPercap, na.rm = TRUE),
    total_population = sum(pop)
  )

print(summary_stats)

# Question 2
highest_lifeExp <- gapminder_cleaned %>%
  filter(year == 2007) %>%
  slice_max(lifeExp, n = 1)

largest_population <- gapminder_cleaned %>%
  filter(year == 2007) %>%
  slice_max(pop, n = 5)

print(highest_lifeExp)
print(largest_population)


# Question 3
ggplot(gapminder_cleaned, aes(x = year, y = lifeExp, color = continent, group = continent)) +
  stat_summary(fun = mean, geom = "line", size = 1.5) +
  labs(title = "Average Life Expectancy Trends by Continent", 
       x = "Year", 
       y = "Life Expectancy") +
  theme_minimal()

# Question 4
ggplot(gapminder_cleaned, aes(x = gdpPercap, y = lifeExp, size = pop, color = continent)) +
  geom_point(alpha = 0.7) +
  scale_x_log10() +
  labs(title = "GDP per Capita vs. Life Expectancy", x = "GDP per Capita (log scale)", y = "Life Expectancy") +
  theme_minimal()

# Question 5
gapminder_cleaned %>%
  filter(year == 2007) %>%
  slice_max(pop, n = 5) %>%
  ggplot(aes(x = reorder(country, -pop), y = pop, fill = country)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 5 Countries by Population (2007)", x = "Country", y = "Population") +
  theme_minimal()

# Question 6
ggplot(gapminder_cleaned, aes(x = continent, y = lifeExp, fill = continent)) +
  geom_boxplot() +
  labs(title = "Distribution of Life Expectancy by Continent", x = "Continent", y = "Life Expectancy") +
  theme_minimal()
