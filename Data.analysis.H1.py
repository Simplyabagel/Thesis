# %%
# Importing cells
import pingouin as pg
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
# %%
# Describing the data
data1 = pd.read_csv(
    '/Users/gautamajay/Documents/University of Amsterdam/Master Thesis/python things/AJ_Data_Try01.csv')
data_demo = pd.read_csv(
    '/Users/gautamajay/Documents/University of Amsterdam/Master Thesis/python things/AJ_Data_Qualtrics_Try1.csv')
print(data1.head())
print(data_demo.head())
print(data1.shape)
print(data_demo.shape)
print(data1.columns)
data1['Participant ID'].value_counts()

# %%
# Plotting a data section example
data1['BR 5'].dropna().astype(float).hist()
plt.title('Distribution of BR 5')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# %% descriptives
print(data1.describe())

# %% Count unique participants
num_participants = data1['Participant ID'].nunique()
print(f"Number of unique participants: {num_participants}")

# %% Surveys administered per participant
survey_counts = data1['Participant ID'].value_counts()
print(survey_counts)
# %% Summary of survey counts
print(survey_counts.describe())

###################### AGE######################
# %% Age statistics
data_demo['Age'] = pd.to_numeric(
    data_demo['Age'], errors='coerce')
average_age = data_demo['Age'].mean()
print(f"Average age: {average_age:.2f}")
age_descriptives = data_demo['Age'].describe()
print(age_descriptives)

#################### GENDER######################
# %% Map gender codes to labels
gender_map = {
    1: "Female",
    2: "Male",
    3: "Nonbinary",
    4: "Agender"
}

data_demo['Gender'] = pd.to_numeric(data_demo['Gender'], errors='coerce')
data_demo['Gender_Label'] = data_demo['Gender'].map(gender_map)

# %% Gender counts
gender_counts = data_demo['Gender_Label'].value_counts()
print("Gender breakdown:")
print(gender_counts)

#################### EDUCATION######################
# %% Check unique raw education values
print(data_demo['Education'].unique())
############ raw coding is 2, 7, and 8. What is 7 and 8?############
# %% Map education codes to labels
education_map = {
    1: "Primary School",
    2: "High School",
    7: "Bachelor",
    8: "Master's",
    5: "PhD"
}

data_demo['Education'] = pd.to_numeric(data_demo['Education'], errors='coerce')
data_demo['Education_Label'] = data_demo['Education'].map(education_map)

# %% Education counts
education_counts = data_demo['Education_Label'].value_counts()
print("Education breakdown:")
print(education_counts)

# %% Preview data1
print(data1.columns)
print(data1.head())

#################### organize data1####################
# Remove these from self-regulation
items_to_remove = ['[ER 2] did you try to change unpleasant feelings',
                   '[ER 3] were you successful in handling your unpleasant emotions',
                   '[ER 5] I did not try to change my emotions because']

# Set CR 5 to missing for value 6
data1['[11_SAQ] CR 5'] = pd.to_numeric(data1['[11_SAQ] CR 5'], errors='coerce')
data1.loc[data1['[11_SAQ] CR 5'] == 6, '[11_SAQ] CR 5'] = pd.NA

# reverse code items
reverse_items = ['ER 6', 'CR 2', '[9_SAQ] CR 3', '[11_SAQ] CR 5',
                 'BR 1', 'BR 3', 'BR 4', 'BR 5']

for item in reverse_items:
    data1[item] = pd.to_numeric(data1[item], errors='coerce')
    data1[item] = 6 - data1[item]  # Reverse code on 1–5 scale

# Emotional (keep ER 1, ER 4, ER 6)
emotional_cols = ['[ER 1] emotion control',
                  '[ER 4] were you successful in handling your unpleasant emotions', 'ER 6']

# Cognitive (CR 1, CR 2, CR 3, CR 4, CR 5, CR 6)
cognitive_cols = ['CR1', 'CR 2', '[9_SAQ] CR 3',
                  '[10_SAQ] CR 4', '[11_SAQ] CR 5', '[12_SAQ] CR 6']

# Behavioral (BR 1–BR 6)
behavioral_cols = ['BR 1', 'BR 2', 'BR 3', 'BR 4', 'BR 5', 'BR 6']

# Final list (excluding removed items)
all_selfreg_cols = emotional_cols + cognitive_cols + behavioral_cols

# Recalculate composite self-regulation
data1['selfreg_mean'] = data1[all_selfreg_cols].mean(axis=1)


# %% Convert all to numeric
for col in all_selfreg_cols:
    data1[col] = pd.to_numeric(data1[col], errors='coerce')

# Aggression columns
aggression_cols = ['AS 1', 'AS 2', 'AS 3', 'AS 4', 'AS 5', 'AS 6']
for col in aggression_cols:
    data1[col] = pd.to_numeric(data1[col], errors='coerce')

# %% Create composite scores
data1['selfreg_mean'] = data1[all_selfreg_cols].mean(axis=1)
data1['aggression_mean'] = data1[aggression_cols].mean(axis=1)
################Cleaned data1######################
# %% Check for duplicates
data1['Participant ID'] = data1['Participant ID'].astype(str).str.strip()
# List of participants to remove
ids_to_remove = ['113293', '104299', '109197', '107743', '109122', '109942', '107531', '112923']

# Filter them out
data1_cleaned = data1[~data1['Participant ID'].isin(ids_to_remove)]
print(
    f"Cleaned data1 sample size: {data1_cleaned['Participant ID'].nunique()}")

# %% excel export
data1_cleaned.to_excel("data1_cleaned.xlsx", index=False)

# %%
data1 = data1_cleaned

######################### trouble shoot#####################

# %% final time conversion
# Convert 'Record Time' to datetime with UTC timezone awareness
data1['Record Time'] = pd.to_datetime(
    data1['Record Time'], format='mixed', utc=True)

# Confirm parsing worked
print(data1['Record Time'].dtype)
print(data1['Record Time'].head())

# Define time zone offsets in hours
timezone_map = {
    104504: 9,
    108828: 2,
    110141: 3,
    # Default is +1 for all others
}

# Adjust Record Time to participant’s local time


def adjust_timezone(row):
    pid = row['Participant ID']
    offset_hours = timezone_map.get(pid, 1)  # Default to +1
    return row['Record Time'] + pd.Timedelta(hours=offset_hours)


# Apply adjustment
data1['Local Record Time'] = data1.apply(adjust_timezone, axis=1)

# Use Local Record Time for survey_date
data1['survey_date'] = data1['Local Record Time'].dt.date

# Sort by Participant ID and Local Record Time (not raw time)
data1 = data1.sort_values(by=['Participant ID', 'Local Record Time'])

# Confirm columns
print(data1.columns)

####################################


########################## Hypothesis 1######################

# 'model'= main model predicting aggression from lagged self-regulation

# 'result'= fitted result of main model

# 'parallel_reverse_model'= parallel reverse model predicting self-regulation

# 'parallel_reverse_model_std' = standardized version of the parallel reverse model

# 'model_std' = standardized main model predicting aggression from lagged self-regulation

# 'result_std' = fitted result of standardized main model

# 'simple_model' = simplified version of the main model without trait self-regulation

# %% Create lagged variables (within day)
data1['selfreg_lag1'] = data1.groupby(['Participant ID', data1['survey_date']])[
    'selfreg_mean'].shift(1)
data1['aggression_lag1'] = data1.groupby(['Participant ID', data1['survey_date']])[
    'aggression_mean'].shift(1)

# %% Person-mean centering of lagged self-regulation
mean_selfreg = data1.groupby('Participant ID')[
    'selfreg_lag1'].transform('mean')
data1['selfreg_lag1_centered'] = data1['selfreg_lag1'] - mean_selfreg

# %% Quick check of lagged data
cols_to_view = ['Participant ID', 'survey_date', 'Record Time',
                'selfreg_mean', 'selfreg_lag1', 'selfreg_lag1_centered',
                'aggression_mean', 'aggression_lag1']

print(data1[cols_to_view].head(10))

# %% Check a random participant’s data
example_id = 107453

data_example = data1[data1['Participant ID'] == example_id][cols_to_view]
print(data_example.head(10))

######################## Double check that there are no lagged values in the first survey of the day###############
# %% check first survey per participant per day
data1['is_first_survey'] = data1.groupby(
    ['Participant ID', 'survey_date']).cumcount() == 0

# %% Check for any first surveys with non-null lag values
invalid_lags = data1[(data1['is_first_survey']) &
                     (data1['selfreg_lag1'].notna())]

print(
    f"Rows where lagged self-regulation appears in first survey of the day: {len(invalid_lags)}")
print(invalid_lags[['Participant ID', 'survey_date',
      'Record Time', 'selfreg_lag1']].head())

# %%

# %% Drop rows with missing values in key variables
model_data = data1.dropna(
    subset=['aggression_mean', 'selfreg_lag1_centered', 'aggression_lag1'])

# %% Fit multilevel (mixed) model
model = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + aggression_lag1",
    data=model_data,
    # Random intercept for each participant
    groups=model_data["Participant ID"]
)

result = model.fit()
print(result.summary())

##################### Ensure Trait self-regulation is not a predictor###########

# %% Compute each participant's average (trait) self-regulation at lag1
trait_selfreg = data1.groupby('Participant ID')[
    'selfreg_lag1'].mean().reset_index()
trait_selfreg.columns = ['Participant ID', 'trait_selfreg']

# %% Merge into main dataset
data1 = pd.merge(data1, trait_selfreg, on='Participant ID', how='left')

# %% Drop rows with missing values in any predictor
model_data = data1.dropna(subset=[
                          'aggression_mean', 'selfreg_lag1_centered', 'aggression_lag1', 'trait_selfreg'])

# %% Fit mixed model
model = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + trait_selfreg + aggression_lag1",
    data=model_data,
    groups=model_data["Participant ID"]
)

result = model.fit()
print(result.summary())


#################### Test Parallel Reverse Model####################

# %% Reverse model with self-regulation at t-1 as a predictor

data1['selfreg_centered'] = data1['selfreg_mean'] - \
    data1.groupby('Participant ID')['selfreg_mean'].transform('mean')


# Drop missing values
parallel_rev_data = data1.dropna(subset=[
    'selfreg_centered', 'aggression_lag1', 'selfreg_lag1_centered', 'trait_selfreg'
])

# Fit parallel reverse model
parallel_reverse_model = smf.mixedlm(
    "selfreg_centered ~ aggression_lag1 + selfreg_lag1_centered + trait_selfreg",
    data=parallel_rev_data,
    groups=parallel_rev_data["Participant ID"]
).fit()

# Show output
print(parallel_reverse_model.summary())

############### UPDATED MODEL COMPARISON####################
# %%Standardize variables


def standardize(series):
    return (series - series.mean()) / series.std()


data1['z_selfreg_centered'] = standardize(data1['selfreg_centered'])
data1['z_selfreg_lag1_centered'] = standardize(data1['selfreg_lag1_centered'])
data1['z_aggression_mean'] = standardize(data1['aggression_mean'])
data1['z_aggression_lag1'] = standardize(data1['aggression_lag1'])
data1['z_trait_selfreg'] = standardize(data1['trait_selfreg'])

############# Standardize Main Model: Self-reg → Aggression####################
# %%
main_model_std_data = data1.dropna(subset=[
    'z_aggression_mean', 'z_selfreg_lag1_centered', 'z_aggression_lag1', 'z_trait_selfreg'])

main_model_std = smf.mixedlm(
    "z_aggression_mean ~ z_selfreg_lag1_centered + z_trait_selfreg + z_aggression_lag1",
    data=main_model_std_data,
    groups=main_model_std_data["Participant ID"]
).fit()

print("Standardized Main Model (Self-reg → Aggression)")
print(main_model_std.summary())

############# Standardized Parallel Reverse Model: Aggression → Self-reg Deficit####################
# %%
reverse_model_std_data = data1.dropna(subset=[
    'z_selfreg_centered', 'z_aggression_lag1', 'z_selfreg_lag1_centered', 'z_trait_selfreg'])

parallel_reverse_model_std = smf.mixedlm(
    "z_selfreg_centered ~ z_aggression_lag1 + z_selfreg_lag1_centered + z_trait_selfreg",
    data=reverse_model_std_data,
    groups=reverse_model_std_data["Participant ID"]
).fit()

print("\nStandardized Parallel Reverse Model (Aggression → Self-reg Deficit)")
print(parallel_reverse_model_std.summary())

############### Compare Effect Sizes####################
# %%
forward_effect = main_model_std.params['z_selfreg_lag1_centered']
reverse_effect = parallel_reverse_model_std.params['z_aggression_lag1']

effect_ratio = abs(forward_effect / reverse_effect)
print(
    f"\nEffect Size Ratio (Self-reg → Agg / Agg → Self-reg Deficit): {effect_ratio:.2f}")

####### EFFECT SIZES GRAPH####################
# %%
forward_effect = main_model_std.params['z_selfreg_lag1_centered']
reverse_effect = parallel_reverse_model_std.params['z_aggression_lag1']

# Data
labels = ['Self-reg → Aggression', 'Aggression → Self-reg']
effects = [forward_effect, reverse_effect]

# Create bar plot
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, effects, color=['gray', 'darkgray'], edgecolor='black')

# Add text labels
for bar, effect in zip(bars, effects):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{effect:.3f}', ha='center', va='bottom', fontsize=10)

# Customize
plt.ylabel('Standardized Coefficient (β)')
plt.title('Comparison of Standardized Effects')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()


################# AIC and BIC####################

# %%
# Fit model WITHOUT trait_selfreg to check AIC/BIC
simple_model = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + aggression_lag1",
    data=model_data_std,
    groups=model_data_std["Participant ID"]
).fit(method='lbfgs')

print(f"AIC: {simple_model.aic:.2f}")
print(f"BIC: {simple_model.bic:.2f}")
######### does not work. random effects error##########


# %%
######### sensitivity analyses################

# %%Fit main model without trait self-reg
model_no_trait = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + aggression_lag1",
    data=model_data,  # Use unstandardized data
    groups=model_data["Participant ID"]
)

result_no_trait = model_no_trait.fit(method='lbfgs')
print("Main Model WITHOUT Trait Self-Reg:")
print(result_no_trait.summary())


# %%highest aggression partipipant excluded
# Find participant with highest average aggression
high_agg_id = data1.groupby('Participant ID')[
    'aggression_mean'].mean().idxmax()
print(f"Excluding participant ID: {high_agg_id}")

# Filter out this participant
filtered_data = model_data[model_data['Participant ID'] != high_agg_id]

# Fit model again
model_filtered = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + trait_selfreg + aggression_lag1",
    data=filtered_data,
    groups=filtered_data["Participant ID"]
)

result_filtered = model_filtered.fit(method='lbfgs')
print("Model After Excluding High-Aggression Participant:")
print(result_filtered.summary())

# %% Residual variance from each model
print(f"Original Model Residual Variance: {result.scale:.4f}")
print(
    f"No Trait Self-Reg Model Residual Variance: {result_no_trait.scale:.4f}")
print(f"Filtered Model Residual Variance: {result_filtered.scale:.4f}")


################## More descriptives######################


######### FIXED SURVEY COMPLETION######################
# %%
all_participants = data1['Participant ID'].unique()

# Count completed surveys where AS 5 is not missing
completed_counts = data1[data1['AS 5'].notna(
)]['Participant ID'].value_counts()

# Step 3: Convert to DataFrame and include everyone
completed_counts = completed_counts.reindex(all_participants, fill_value=0)

# final summary
print(completed_counts.describe())

# %% Self-regulation and Aggression
print(data1['selfreg_mean'].describe())
print(data1['aggression_mean'].describe())

# %% Within-Person Variability
variability = data1.groupby('Participant ID')['selfreg_mean'].std()
print(variability.describe())


############ UPDATED RESULTS##############


# %% Main and parallel reverse model (unstandardized)
print("Main Model: Self-reg → Aggression")
print(result.summary())

print("\nParallel Reverse Model: Aggression → Self-reg Deficit")
print(parallel_reverse_model.summary())  # <- replaces reverse_result

# %% Standardized models
print("\nStandardized Model: Self-reg → Aggression")
print(result_std.summary())

print("\nStandardized Parallel Reverse Model: Aggression → Self-reg Deficit")
print(parallel_reverse_model_std.summary())  # <- replaces reverse_result_std

# %% Within-person variability
within_person_sd = data1.groupby('Participant ID')['selfreg_mean'].std()

print("\nWithin-person SD (self-regulation):")
print(within_person_sd.describe())

##################### Participant check######################
# %% participant check
# Total participants
total_participants = data1['Participant ID'].nunique()

# Participants used in model (no missing data)
model_participants = model_data['Participant ID'].nunique()

print(f"Total participants: {total_participants}")
print(f"Participants in model: {model_participants}")
print(f"Excluded participants: {total_participants - model_participants}")

# Participants included in the model
included_ids = set(model_data['Participant ID'].unique())

# All participants
all_ids = set(data1['Participant ID'].unique())

# Participants excluded
excluded_ids = all_ids - included_ids
print(f"Excluded participant IDs: {sorted(excluded_ids)}")
print(f"Number of excluded participants: {len(excluded_ids)}")

# Dataset with only excluded participants
excluded_data = data1[data1['Participant ID'].isin(excluded_ids)]

# Survey count per excluded participant
survey_counts_excluded = excluded_data['Participant ID'].value_counts()
print("Survey counts for excluded participants:")
print(survey_counts_excluded)

# Check missingness in key variables
missing_summary = excluded_data[['selfreg_lag1_centered',
                                 'aggression_lag1', 'trait_selfreg', 'aggression_mean']].isnull().sum()
print("\nMissing values in key variables (excluded participants):")
print(missing_summary)

# %%Check missing variables per excluded participant
excluded_data['missing_vars'] = excluded_data[['trait_selfreg',
                                               'selfreg_lag1_centered', 'aggression_lag1']].isnull().sum(axis=1)

# Participants with all 3 missing
fully_missing = excluded_data[excluded_data['missing_vars']
                              == 3]['Participant ID'].unique()
print(
    f"Participants with all key lag variables missing: {sorted(fully_missing)}")

##################### GRAPHS######################

# %% Predicted aggression from main model (Predicted vs. Actual)
model_data['predicted_aggression'] = result.predict()

plt.figure(figsize=(7, 5))
plt.scatter(model_data['predicted_aggression'],
            model_data['aggression_mean'], alpha=0.5, edgecolor='black')
plt.title("Predicted vs. Actual Aggression Scores")
plt.xlabel("Predicted Aggression")
plt.ylabel("Actual Aggression")
plt.plot([model_data['predicted_aggression'].min(), model_data['predicted_aggression'].max()],
         [model_data['predicted_aggression'].min(
         ), model_data['predicted_aggression'].max()],
         'k--', lw=1)  # Line of perfect prediction
plt.tight_layout()
plt.show()


# %% aggression by self-reg level (low vs high)
# Group by low, average, high self-regulation
model_data['selfreg_level'] = pd.qcut(
    model_data['selfreg_lag1_centered'], q=3, labels=['Low', 'Average', 'High'])

# Average aggression for each group
agg_by_selfreg = model_data.groupby('selfreg_level')['aggression_mean'].mean()

# Plot
plt.figure(figsize=(6, 5))
agg_by_selfreg.plot(kind='bar', color='gray', edgecolor='black')
plt.title("Aggression by Self-Regulation Level at t-1")
plt.xlabel("Self-Regulation Level (Lagged)")
plt.ylabel("Mean Aggression")
plt.tight_layout()
plt.show()

######### Within person variability######################

# %% Within-person variability
# Calculate within-person SD of self-regulation
within_person_sd = data1.groupby('Participant ID')['selfreg_mean'].std()

# View overall summary
print("Within-person SD (self-regulation):")
print(within_person_sd.describe())


plt.figure(figsize=(6, 4))
within_person_sd.plot(kind='hist', bins=15, edgecolor='black')
plt.title("Distribution of Within-Person SD in Self-Regulation")
plt.xlabel("Standard Deviation")
plt.ylabel("Number of Participants")
plt.tight_layout()
plt.show()


# %% CORRECTED Standardized coefficients


# Standardized coefficients
effects = {
    "Self-Reg → Aggression": -0.024,
    "Aggression → Self-Reg": -0.050
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 5))

# Bar plot
bars = ax.bar(effects.keys(), effects.values(),
              color='gray', edgecolor='black')

# Horizontal line at zero
ax.axhline(0, color='black', linewidth=0.8)

# Add numeric coefficient labels
for bar in bars:
    height = bar.get_height()
    offset = 0.01 if height > 0 else -0.01
    ax.text(bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha='center',
            va='bottom' if height > 0 else 'top')

# Extend y-axis range for visual space
y_min = min(effects.values()) - 0.03
y_max = max(effects.values()) + 0.03
ax.set_ylim(y_min, y_max)

# Labels and styling
ax.set_title("Standardized Effect Sizes for Lagged Models")
ax.set_ylabel("Standardized Coefficient (β)")
plt.tight_layout()
plt.show()


# %% Aggression distribution

plt.figure(figsize=(6, 5))
model_data['aggression_mean'].plot(kind='hist', bins=20, edgecolor='black')
plt.title("Distribution of Aggression Scores")
plt.xlabel("Aggression (Composite Score)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# %% Descriptive stats
print("Age:")
print(data_demo['Age'].describe())

print("Self-Regulation Composite:")
print(data1['selfreg_mean'].describe())

print("Aggression Composite:")
print(data1['aggression_mean'].describe())

################# ASSUMPTION CHECKS######################

# %% Main model Assumption checks


# Residual Normality and Linearity
residuals = result.resid  # For main model

# Histogram
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, bins=20, edgecolor='black')
plt.title("Histogram of Model Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()


# %% Homoscedasticity

fitted = result.fittedvalues

plt.figure(figsize=(6, 4))
plt.scatter(fitted, residuals, alpha=0.4, edgecolor='black')
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals vs. Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# %% PARALLEL REVERSE MODEL ASSUMPTION CHECKS
# Extract residuals and fitted values from the parallel reverse model
reverse_resid = parallel_reverse_model_std.resid
reverse_fitted = parallel_reverse_model_std.fittedvalues

# Histogram of residuals
plt.figure(figsize=(6, 4))
sns.histplot(reverse_resid, kde=True, bins=20, edgecolor='black')
plt.title("Histogram of Residuals (Parallel Reverse Model)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Q–Q plot
stats.probplot(reverse_resid, dist="norm", plot=plt)
plt.title("Q–Q Plot of Residuals (Parallel Reverse Model)")
plt.tight_layout()
plt.show()

# Residuals vs. fitted values
plt.figure(figsize=(6, 4))
plt.scatter(reverse_fitted, reverse_resid, alpha=0.4, edgecolor='black')
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals vs. Fitted Values (Parallel Reverse Model)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


############################# POST HOC ANALYSIS######################


# %% Main model with high-aggression participants only >4 on an item  (AS 1-6)
aggression_items = ['AS 1', 'AS 2', 'AS 3', 'AS 4', 'AS 5', 'AS 6']

# Filter rows where any aggression item ≥ 4
high_agg_rows = data1[aggression_items].ge(4).any(axis=1)

# Get Participant IDs
high_agg_participants = data1.loc[high_agg_rows, 'Participant ID'].unique()
high_agg_data = data1[data1['Participant ID'].isin(high_agg_participants)]

model_high_agg = high_agg_data.dropna(subset=[
    'aggression_mean', 'selfreg_lag1_centered', 'aggression_lag1', 'trait_selfreg'
])

# Fit model
model_high = smf.mixedlm(
    "aggression_mean ~ selfreg_lag1_centered + trait_selfreg + aggression_lag1",
    data=model_high_agg,
    groups=model_high_agg["Participant ID"]
).fit()

print(model_high.summary())

# %%Main model where participants scored at least a 4 on one AS item as an outcome

# Mark if any aggression item at that survey was scored ≥ 4
aggression_items = ['AS 1', 'AS 2', 'AS 3', 'AS 4', 'AS 5', 'AS 6']
data1['high_aggression'] = data1[aggression_items].ge(
    4).any(axis=1).astype(int)

logit_data = data1.dropna(subset=[
    'high_aggression', 'selfreg_lag1_centered', 'aggression_lag1', 'trait_selfreg'
])

logit_model = smf.glm(
    formula="high_aggression ~ selfreg_lag1_centered + trait_selfreg + aggression_lag1",
    data=logit_data,
    family=sm.families.Binomial()
).fit()

print(logit_model.summary())

np.exp(logit_model.params)

odds_ratios = np.exp([1.030, 0.789, 3.190])
print(odds_ratios)


# %%Conccurent self-regulation and aggression
# Person-mean centered self-reg at time t
data1['selfreg_centered'] = data1['selfreg_mean'] - \
    data1.groupby('Participant ID')['selfreg_mean'].transform('mean')

# Drop missing and fit the model
same_time_centered = data1.dropna(
    subset=['aggression_mean', 'selfreg_centered', 'trait_selfreg'])

model_centered = smf.mixedlm(
    "aggression_mean ~ selfreg_centered + trait_selfreg",
    data=same_time_centered,
    groups=same_time_centered["Participant ID"]
).fit()

print(model_centered.summary())


#################### CRonbach's alpha######################
# %%

# Combine emotional, cognitive, and behavioral columns
selfreg_items = emotional_cols + cognitive_cols + behavioral_cols
aggression_items = ['AS 1', 'AS 2', 'AS 3', 'AS 4', 'AS 5', 'AS 6']

# Self-regulation alpha
alpha_selfreg = pg.cronbach_alpha(data1[selfreg_items])[0]
print(f"Cronbach’s alpha for self-regulation composite: {alpha_selfreg:.3f}")

# Aggression alpha
alpha_agg = pg.cronbach_alpha(data1[aggression_items])[0]
print(f"Cronbach’s alpha for aggression composite: {alpha_agg:.3f}")


######################### check new demographics of n=162####################
# %%participant removal
data_demo = data_demo.rename(columns={"Q36": "Participant ID"})

# Filter to participants who completed at least one survey (AS 5 not missing)
valid_ids = completed_counts[completed_counts > 0].index

# Filter data1 to include only these participants
data1_valid = data1[data1['Participant ID'].isin(valid_ids)]
print(
    f"Number of participants who completed at least one survey: {n_valid_participants}")

# %%
# Find participants with zero completed surveys
no_survey_ids = completed_counts[completed_counts == 0].index.tolist()
print("Participant(s) with 0 surveys:", no_survey_ids)
# Remove those participants from data_demo
data_demo_cleaned = data_demo[~data_demo['Participant ID'].isin(no_survey_ids)]
print(
    f"New sample size in data_demo: {data_demo_cleaned['Participant ID'].nunique()}")

# %% total demo
n_total_demo = data_demo['Participant ID'].nunique()
print(f"Number of participants in data_demo before removal: {n_total_demo}")

########### CROSS Reference IDs  ##############
# %%Get unique IDs from both datasets
# Convert both sets of IDs
ids_data1 = set(data1['Participant ID'].astype(str).str.strip().unique())
ids_data_demo = set(
    data_demo['Participant ID'].astype(str).str.strip().unique())

# Find IDs in data1 but missing in data_demo
missing_in_demo = ids_data1 - ids_data_demo

# Show results
print(
    f"Participants in data1 but missing from data_demo: {sorted(missing_in_demo)}")
print(f"Total missing: {len(missing_in_demo)}")


# %%Reverse check
# Find IDs in data_demo but not in data1
missing_in_data1 = ids_data_demo - ids_data1

# Show results
print(
    f"Participants in data_demo but missing from data1: {sorted(missing_in_data1)}")
print(f"Total missing: {len(missing_in_data1)}")
# %%
# Count how many times each ID appears
id_counts = data_demo['Participant ID'].value_counts()

# Filter for IDs that appear more than once
duplicate_ids = id_counts[id_counts > 1]
print("Duplicate Participant IDs:")
print(duplicate_ids)

#####################Fix data demo#######################

# %%
# Make sure IDs are strings and remove extra spaces
data_demo['Participant ID'] = data_demo['Participant ID'].astype(str).str.strip()

# %%
# Correct typos
data_demo.loc[data_demo['Participant ID'] == '101144', 'Participant ID'] = '110144'
data_demo.loc[data_demo['Participant ID'] == '110438', 'Participant ID'] = '111438'

# %%
# Define IDs to remove
ids_to_remove = ['109197', '109122', '107531', '?']

# Filter out those rows
data_demo_cleaned = data_demo[~data_demo['Participant ID'].isin(ids_to_remove)]

# %%
print(f"Cleaned data_demo sample size: {data_demo_cleaned['Participant ID'].nunique()}")

###################CREATE NEW DATA DEMO######################
# %%
data_demo_cleaned2 = pd.read_csv('/Users/gautamajay/Documents/University of Amsterdam/Master Thesis/python things/AJ_Data_Qualtrics.cleaned.csv')


# %%
# Load your cleaned demographics CSV using the correct delimiter
data_demo_cleaned2 = pd.read_csv("AJ_Data_Qualtrics.cleaned.csv", sep=';')

# Clean up column names
data_demo_cleaned2.columns = data_demo_cleaned2.columns.str.strip()

# Convert relevant columns to numeric
data_demo_cleaned2['Age'] = pd.to_numeric(data_demo_cleaned2['Age'], errors='coerce')
data_demo_cleaned2['Gender'] = pd.to_numeric(data_demo_cleaned2['Gender'], errors='coerce')
data_demo_cleaned2['Education'] = pd.to_numeric(data_demo_cleaned2['Education'], errors='coerce')

# Descriptive statistics for Age
age_desc = data_demo_cleaned2['Age'].describe()[['mean', 'std', 'min', 'max']]
print("Age Descriptives:")
print(age_desc)

# Gender breakdown
gender_map = {
    1: "Female",
    2: "Male",
    3: "Nonbinary",
    4: "Agender"
}
data_demo_cleaned2['Gender_Label'] = data_demo_cleaned2['Gender'].map(gender_map)
gender_counts = data_demo_cleaned2['Gender_Label'].value_counts()
print("\nGender Breakdown:")
print(gender_counts)

# Education breakdown
education_map = {
    1: "Primary School",
    2: "High School",
    7: "Bachelor",
    8: "Master's",
    5: "PhD"
}
data_demo_cleaned2['Education_Label'] = data_demo_cleaned2['Education'].map(education_map)
education_counts = data_demo_cleaned2['Education_Label'].value_counts()
print("\nEducation Breakdown:")
print(education_counts)

#############Clean main data1#################
# %%
data1['Participant ID'] = data1['Participant ID'].astype(str).str.strip()
# List of participants to remove
ids_to_remove = ['113293', '104299', '109197', '107743', '109122', '109942', '107531', '112923']

# Filter them out
data1_cleaned = data1[~data1['Participant ID'].isin(ids_to_remove)]
print(
    f"Cleaned data1 sample size: {data1_cleaned['Participant ID'].nunique()}")


# %%Get unique IDs from both datasets
data1_cleaned['Participant ID'] = data1_cleaned['Participant ID'].astype(str).str.strip()
data_demo_cleaned['Participant ID'] = data_demo_cleaned['Participant ID'].astype(str).str.strip()

# Get sets of unique IDs
ids_data1_cleaned = set(data1_cleaned['Participant ID'].unique())
ids_demo_cleaned = set(data_demo_cleaned['Participant ID'].unique())

# Find participants in cleaned data1 but missing from cleaned demo
missing_demo_ids = ids_data1_cleaned - ids_demo_cleaned

print("Participants in data1_cleaned but missing from data_demo_cleaned:")
print(sorted(missing_demo_ids))

################Data1_cleaned analysis rerun##################

# %%DESCRIPTIVES
# Make sure Participant ID is string and stripped
data1_cleaned['Participant ID'] = data1_cleaned['Participant ID'].astype(str).str.strip()

# Count valid surveys per participant (using a reliable column like 'AS 5')
survey_counts = data1_cleaned[data1_cleaned['AS 5'].notna()]['Participant ID'].value_counts()

# Create a participant-level summary for self-regulation and aggression
summary = data1_cleaned.groupby('Participant ID').agg({
    'selfreg_mean': 'mean',
    'aggression_mean': 'mean'
}).reset_index()

# Add survey counts to the summary
summary['Surveys Completed'] = summary['Participant ID'].map(survey_counts).fillna(0)

# Describe key variables
desc_stats = summary[['Surveys Completed', 'selfreg_mean', 'aggression_mean']].describe().T
desc_stats = desc_stats[['mean', 'std', 'min', 'max']]
desc_stats.columns = ['M', 'SD', 'Min', 'Max']

print(desc_stats)
# %%
n_participants = summary['Participant ID'].nunique()
print(f"Number of participants included in summary: {n_participants}")




# %%
# Optionally overwrite
data_demo = data_demo_cleaned2
# %%Optionally overwrite
data1 = data1_cleaned

# %%
