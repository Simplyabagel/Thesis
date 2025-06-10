# %%
# Importing cells
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# %% Surveys per participant
survey_counts = data1['Participant ID'].value_counts()
print(survey_counts)
# %% Summary of survey counts
print(survey_counts.describe())

# %% Age statistics
data_demo['Age'] = pd.to_numeric(
    data_demo['Age'], errors='coerce')
average_age = data_demo['Age'].mean()
print(f"Average age: {average_age:.2f}")
age_descriptives = data_demo['Age'].describe()
print(age_descriptives)

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


#################### organize data1 ####################

# Remove unnecessary self-regulation items
items_to_remove = ['[ER 2] did you try to change unpleasant feelings',
                   '[ER 3] were you successful in handling your unpleasant emotions',
                   '[ER 5] I did not try to change my emotions because']


# Set CR 5 to missing for value 6
data1['[11_SAQ] CR 5'] = pd.to_numeric(data1['[11_SAQ] CR 5'], errors='coerce')
data1.loc[data1['[11_SAQ] CR 5'] == 6, '[11_SAQ] CR 5'] = pd.NA

# Reverse code  items
reverse_items = ['ER 6', 'CR 2', '[9_SAQ] CR 3', '[11_SAQ] CR 5',
                 'BR 1', 'BR 3', 'BR 4', 'BR 5']
for item in reverse_items:
    data1[item] = pd.to_numeric(data1[item], errors='coerce')
    data1[item] = 6 - data1[item]

# Define column groups
emotional_cols = ['[ER 1] emotion control',
                  '[ER 4] were you successful in handling your unpleasant emotions', 'ER 6']
cognitive_cols = ['CR1', 'CR 2', '[9_SAQ] CR 3',
                  '[10_SAQ] CR 4', '[11_SAQ] CR 5', '[12_SAQ] CR 6']
behavioral_cols = ['BR 1', 'BR 2', 'BR 3', 'BR 4', 'BR 5', 'BR 6']
all_selfreg_cols = emotional_cols + cognitive_cols + behavioral_cols

# Convert all self-regulation items to numeric
for col in all_selfreg_cols:
    data1[col] = pd.to_numeric(data1[col], errors='coerce')

# Convert aggression items to numeric
aggression_items = ['AS 1', 'AS 2', 'AS 3', 'AS 4', 'AS 5', 'AS 6']
for item in aggression_items:
    data1[item] = pd.to_numeric(data1[item], errors='coerce')

# Composite scores
data1['emotional_regulation_mean'] = data1[emotional_cols].mean(axis=1)
data1['behavioral_regulation_mean'] = data1[behavioral_cols].mean(axis=1)
data1['selfreg_mean'] = data1[all_selfreg_cols].mean(axis=1)
data1['aggression_mean'] = data1[aggression_items].mean(axis=1)

# Verbal aggression (AS 2 and AS 3 only)
verbal_aggression_cols = ['AS 2', 'AS 3']
data1['verbal_aggression_mean'] = data1[verbal_aggression_cols].mean(axis=1)

# Physical aggression (AS 6 only)
data1['physical_aggression'] = data1['AS 6']


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
# %% Check for duplicates
data1['Participant ID'] = data1['Participant ID'].astype(str).str.strip()
# List of participants to remove
ids_to_remove = ['113293', '104299', '109197', '107743', '109122', '109942', '107531', '112923']

# Filter them out
data1_cleaned = data1[~data1['Participant ID'].isin(ids_to_remove)]
print(
    f"Cleaned data1 sample size: {data1_cleaned['Participant ID'].nunique()}")
data1 = data1_cleaned
###################### Hypothesis 2####################

# %% Hypothesis 2a emotional regulation and verbal aggression


# Filter data for H2a — only non-missing values for relevant columns
data_h2a = data1.dropna(subset=['verbal_aggression_mean',
                        'emotional_regulation_mean', 'behavioral_regulation_mean'])

# Model H2a: Verbal Aggression ~ Emotional SR (controlling for Behavioral SR)
model_h2a = smf.mixedlm("verbal_aggression_mean ~ emotional_regulation_mean + behavioral_regulation_mean",
                        data_h2a, groups=data_h2a["Participant ID"])
results_h2a = model_h2a.fit()
print(results_h2a.summary())

##### Both ER and BR are significant predictors of verbal aggression###

# %%
### which predictor is stronger?###
# Create standardized (z-scored) predictors
data_h2a['emotional_z'] = (data_h2a['emotional_regulation_mean'] -
                           data_h2a['emotional_regulation_mean'].mean()) / data_h2a['emotional_regulation_mean'].std()
data_h2a['behavioral_z'] = (data_h2a['behavioral_regulation_mean'] -
                            data_h2a['behavioral_regulation_mean'].mean()) / data_h2a['behavioral_regulation_mean'].std()

# Standardized model
model_h2a_std = smf.mixedlm("verbal_aggression_mean ~ emotional_z + behavioral_z",
                            data_h2a, groups=data_h2a["Participant ID"])
results_h2a_std = model_h2a_std.fit()
print(results_h2a_std.summary())

### BR is a stronger predictor of verbal aggression###

############# H2a Graphs ######################
# %% scatter plot with regression line ER and VA


sns.set(style="whitegrid")  # Start with clean grid
sns.set_context("talk")     # Adjust font sizes to approx. APA style

# Create plot
sns.lmplot(x='emotional_regulation_mean', y='verbal_aggression_mean', data=data_h2a,
           scatter_kws={'alpha': 0.3, 'color': 'gray'}, line_kws={'color': 'black'})

# Remove gridlines for APA
plt.grid(False)
plt.title("Emotional Self-Regulation Predicting Verbal Aggression", fontsize=14)
plt.xlabel("Emotional Self-Regulation (Mean)", fontsize=12)
plt.ylabel("Verbal Aggression (Mean)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# %% bar plot comparing ER and BR
# Extract standardized coefficients
standardized_emotional = results_h2a_std.params['emotional_z']
standardized_behavioral = results_h2a_std.params['behavioral_z']

# Create DataFrame for plotting
coef_df = pd.DataFrame({
    'Predictor': ['Emotional SR', 'Behavioral SR'],
    'Standardized Coefficient': [standardized_emotional, standardized_behavioral]
})

print(coef_df)

sns.set(style="white")
sns.set_context("talk")

# Bar plot
sns.barplot(x='Predictor', y='Standardized Coefficient',
            data=coef_df, color='gray', edgecolor='black')

plt.title("Standardized Effects on Verbal Aggression", fontsize=14)
plt.ylabel("Standardized Beta", fontsize=12)
plt.xlabel("")  # No need for xlabel here
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()


####################### Hypothesis 2b######################
# %% Hypothesis 2b behavioral regulation and physical aggression

data_h2b = data1.dropna(subset=[
                        'physical_aggression', 'emotional_regulation_mean', 'behavioral_regulation_mean'])

model_h2b = smf.mixedlm("physical_aggression ~ emotional_regulation_mean + behavioral_regulation_mean",
                        data_h2b, groups=data_h2b["Participant ID"])
results_h2b = model_h2b.fit()
print(results_h2b.summary())

########## Both ER and BR are sig predictors#########
# %% which predictor is stronger
# Standardize within the filtered H2b dataset
data_h2b['emotional_z'] = (data_h2b['emotional_regulation_mean'] -
                           data_h2b['emotional_regulation_mean'].mean()) / data_h2b['emotional_regulation_mean'].std()
data_h2b['behavioral_z'] = (data_h2b['behavioral_regulation_mean'] -
                            data_h2b['behavioral_regulation_mean'].mean()) / data_h2b['behavioral_regulation_mean'].std()


model_h2b_std = smf.mixedlm("physical_aggression ~ emotional_z + behavioral_z",
                            data_h2b, groups=data_h2b["Participant ID"])
results_h2b_std = model_h2b_std.fit()
print(results_h2b_std.summary())

########## H2b Graphs####################

# %% bar plot comparing ER and BR

# Build DataFrame with standardized betas from results_h2b_std
coef_df_h2b = pd.DataFrame({
    'Predictor': ['Emotional SR', 'Behavioral SR'],
    'Standardized Coefficient': [results_h2b_std.params['emotional_z'], results_h2b_std.params['behavioral_z']]
})

print(coef_df_h2b)


sns.set(style="white")
sns.set_context("talk")

sns.barplot(x='Predictor', y='Standardized Coefficient',
            data=coef_df_h2b, color='gray', edgecolor='black')

plt.title("Standardized Effects on Physical Aggression", fontsize=14)
plt.ylabel("Standardized Beta", fontsize=12)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()


# %% Regressiong plot for BR and PA


sns.set(style="white")
sns.set_context("talk")

# Scatter plot with regression line
sns.lmplot(x='behavioral_regulation_mean', y='physical_aggression', data=data_h2b,
           scatter_kws={'alpha': 0.3, 'color': 'gray'}, line_kws={'color': 'black'})

plt.title("Behavioral Self-Regulation Predicting Physical Aggression", fontsize=14)
plt.xlabel("Behavioral Self-Regulation (Mean)", fontsize=12)
plt.ylabel("Physical Aggression", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()

############# Sensitivity analysis####################
# %%Total number of surveys per participant
total_surveys = data1.groupby("Participant ID").size()

# Number of valid (non-missing) verbal aggression entries per participant
valid_surveys = data1.groupby("Participant ID")[
    'verbal_aggression_mean'].apply(lambda x: x.notna().sum())

# Completion rate
completion_rate = (valid_surveys / total_surveys).reset_index()
completion_rate.columns = ['Participant ID', 'completion_rate']

# Merge completion rate back into your dataset
data1 = data1.merge(completion_rate, on='Participant ID')

# %%Keep only participants who completed at least 50% of their surveys
data_sensitivity = data1[data1['completion_rate'] >= 0.5]

data_h2a_sens = data_sensitivity.dropna(
    subset=['verbal_aggression_mean', 'emotional_regulation_mean', 'behavioral_regulation_mean'])

model_h2a_sens = smf.mixedlm("verbal_aggression_mean ~ emotional_regulation_mean + behavioral_regulation_mean",
                             data_h2a_sens, groups=data_h2a_sens["Participant ID"])
results_h2a_sens = model_h2a_sens.fit()
print(results_h2a_sens.summary())

# %% Hypothesis 2b sensitivity analysis
data_h2b_sens = data_sensitivity.dropna(
    subset=['physical_aggression', 'emotional_regulation_mean', 'behavioral_regulation_mean'])

model_h2b_sens = smf.mixedlm("physical_aggression ~ emotional_regulation_mean + behavioral_regulation_mean",
                             data_h2b_sens, groups=data_h2b_sens["Participant ID"])
results_h2b_sens = model_h2b_sens.fit()
print(results_h2b_sens.summary())


######################### RESULTS######################
# %% H2a results
print(results_h2a.summary())

# %% H2a standardized results
print(results_h2a_std.summary())


# %% H2b results
print(results_h2b.summary())

# %% H2b standardized results
print(results_h2b_std.summary())


#################### H2a and H2b BAR PLOTS######################


# %% H2a bar plot
coef_df_h2a = pd.DataFrame({
    'Predictor': ['Emotional SR', 'Behavioral SR'],
    'Standardized Coefficient': [
        results_h2a_std.params['emotional_z'],
        results_h2a_std.params['behavioral_z']
    ]
})

sns.set(style="white")
sns.set_context("talk")

plt.figure(figsize=(6, 4))
ax = sns.barplot(x='Predictor', y='Standardized Coefficient', data=coef_df_h2a,
                 color='gray', edgecolor='black')

# Set y-axis limit slightly below the lowest bar
ymin = coef_df_h2a['Standardized Coefficient'].min() - 0.02
plt.ylim(ymin, 0)

# Annotate coefficient values on bars
for i, row in coef_df_h2a.iterrows():
    ax.text(i, row['Standardized Coefficient'] - 0.005, f"{row['Standardized Coefficient']:.3f}",
            ha='center', va='top', color='black', fontsize=12)

plt.title("Standardized Effects on Verbal Aggression", fontsize=14)
plt.ylabel("Standardized Beta", fontsize=12)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()


# %% H2b bar plot

# Extract standardized coefficients from H2b
coef_df_h2b = pd.DataFrame({
    'Predictor': ['Emotional SR', 'Behavioral SR'],
    'Standardized Coefficient': [
        results_h2b_std.params['emotional_z'],
        results_h2b_std.params['behavioral_z']
    ]
})
plt.figure(figsize=(6, 4))
ax = sns.barplot(x='Predictor', y='Standardized Coefficient', data=coef_df_h2b,
                 color='gray', edgecolor='black')

# Set y-axis limit slightly below the lowest bar
ymin = coef_df_h2b['Standardized Coefficient'].min() - 0.02
plt.ylim(ymin, 0)

# Annotate coefficient values on bars
for i, row in coef_df_h2b.iterrows():
    ax.text(i, row['Standardized Coefficient'] - 0.005, f"{row['Standardized Coefficient']:.3f}",
            ha='center', va='top', color='black', fontsize=12)

plt.title("Standardized Effects on Physical Aggression", fontsize=14)
plt.ylabel("Standardized Beta", fontsize=12)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()


####################### ASSUMPTION CHECKS######################

# %%H2a assumption checks


plt.hist(results_h2a.resid, bins=30, color='gray', edgecolor='black')
plt.title("Residuals Histogram – H2a")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


stats.probplot(results_h2a.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot – H2a Residuals")
plt.tight_layout()
plt.show()

fitted = results_h2a.fittedvalues
residuals = results_h2a.resid

plt.scatter(fitted, residuals, alpha=0.3, color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("Residuals vs. Fitted – H2a")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# Residuals vs. Emotional SR
plt.scatter(data_h2a['emotional_regulation_mean'],
            residuals, alpha=0.3, color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("Residuals vs. Emotional SR – H2a")
plt.xlabel("Emotional SR (Mean)")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


# %%H2b assumption checks
plt.hist(results_h2b.resid, bins=30, color='gray', edgecolor='black')
plt.title("Residuals Histogram – H2b")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


stats.probplot(results_h2b.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot – H2b Residuals")
plt.tight_layout()
plt.show()

fitted_b = results_h2b.fittedvalues
residuals_b = results_h2b.resid

plt.scatter(fitted_b, residuals_b, alpha=0.3, color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("Residuals vs. Fitted – H2b")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

plt.scatter(data_h2b['behavioral_regulation_mean'],
            residuals_b, alpha=0.3, color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("Residuals vs. Behavioral SR – H2b")
plt.xlabel("Behavioral SR (Mean)")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

####################### Exploratory analysis######################
# %% data prep

# Cognitive items
cognitive_cols = ['CR1', 'CR 2', '[9_SAQ] CR 3',
                  '[10_SAQ] CR 4', '[11_SAQ] CR 5', '[12_SAQ] CR 6']
for col in cognitive_cols:
    data1[col] = pd.to_numeric(data1[col], errors='coerce')
data1['cognitive_regulation_mean'] = data1[cognitive_cols].mean(axis=1)

# Data with all three SR dimensions
# Drop rows with any missing values for aggression or SR dimensions
data_explore = data1.dropna(subset=[
    'verbal_aggression_mean',
    'physical_aggression',
    'emotional_regulation_mean',
    'behavioral_regulation_mean',
    'cognitive_regulation_mean'  # You need this column calculated
])
# Create cognitive regulation score
cognitive_cols = ['CR1', 'CR 2', '[9_SAQ] CR 3',
                  '[10_SAQ] CR 4', '[11_SAQ] CR 5', '[12_SAQ] CR 6']
data1['cognitive_regulation_mean'] = data1[cognitive_cols].mean(axis=1)

# Standardize SR predictors
data_explore['emotional_z'] = (data_explore['emotional_regulation_mean'] -
                               data_explore['emotional_regulation_mean'].mean()) / data_explore['emotional_regulation_mean'].std()
data_explore['behavioral_z'] = (data_explore['behavioral_regulation_mean'] -
                                data_explore['behavioral_regulation_mean'].mean()) / data_explore['behavioral_regulation_mean'].std()
data_explore['cognitive_z'] = (data_explore['cognitive_regulation_mean'] -
                               data_explore['cognitive_regulation_mean'].mean()) / data_explore['cognitive_regulation_mean'].std()

# %% Verbal aggression model


model_verbal_explore = smf.mixedlm("verbal_aggression_mean ~ emotional_z + behavioral_z + cognitive_z",
                                   data_explore, groups=data_explore["Participant ID"])
results_verbal_explore = model_verbal_explore.fit()
print(results_verbal_explore.summary())

# %% Physical aggression model
model_physical_explore = smf.mixedlm("physical_aggression ~ emotional_z + behavioral_z + cognitive_z",
                                     data_explore, groups=data_explore["Participant ID"])
results_physical_explore = model_physical_explore.fit()
print(results_physical_explore.summary())


# %%
