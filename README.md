[![ballet](https://img.shields.io/static/v1?label=built%20with&message=ballet&color=FCDD35)](https://ballet.github.io)
<a href="https://mybinder.org/v2/gh/HDI-Project/ballet-fragile-families/master?urlpath=lab" target="_blank" rel="nofollow"><img src="https://ballet.github.io/ballet/_static/launch-assemble.svg" style="max-width:100%;"></a>
[![slack](https://img.shields.io/badge/Slack-Join%20Channel-36C5F0?logo=slack)](https://join.slack.com/share/zt-r6u5ohtf-V40eunINB56ZD2Qd3b1L1w)
[![Join the chat at https://gitter.im/ballet-project/fragile-families](https://badges.gitter.im/ballet-project/fragile-families.svg)](https://gitter.im/ballet-project/fragile-families?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Fragile Families Collaboration

This is a collaborative predictive modeling project built on the [ballet framework](https://ballet.github.io).

The Fragile Families Challenge (FFC) is a recent attempt to better connect to the social science research community to new tools in data science and machine learning. This challenge aimed to spur the development of predictive models for life outcomes from data collected as part of the Fragile Families and Child Wellbeing Study (FFCWS), which collects detailed longitudinal records on a set of disadvantaged children and their families. Organizers released anonymized and merged data on a set of 4,242 families with data collected from the birth of the child until age 9. Participants in the challenge were then tasked with predicting six life outcomes of the child or family at age 15: child grade point average, child grit, household eviction, household material hardship, primary caregiver layoff, and primary caregiver participation in job training. The FFC was run over a four month period in 2017 and received 160 submissions from social scientists, machine learning practitioners, students, and others.

In this project, we ask, by collaborating rather than competing, can we develop impactful solutions to the FFC? Participants in the FFC were competing against each other to produce the best performing models, at the expense of collaboration across teams.

*Your task is to create and submit feature definitions to our shared project that help us in predicting these key life outcomes.*

## Join the collaboration

Are you interested in joining the collaboration?

1. [Apply for access to the dataset and then register yourself with us](#apply-for-access-and-registration).
1. Read/skim the [Ballet Contributor Guide](https://ballet.github.io/ballet/contributor_guide.html).
1. Read/skim the [Ballet Feature Engineering Guide](https://ballet.github.io/ballet/feature_engineering_guide.html).
1. Learn more about the Fragile Families dataset.
    1. Read/skim the [data documentation](#data-documentation).
    1. Skim [additional resources](#additional-resources).
1. Browse the currently accepted features in the contributed features directory ([`src/fragile_families/features/contrib`](src/fragile_families/features/contrib)).
1. Launch an interactive Jupyter Lab session to hack on this repository: <a href="https://mybinder.org/v2/gh/HDI-Project/ballet-fragile-families/master?urlpath=lab" target="_blank" rel="nofollow" ><img src="https://ballet.github.io/ballet/_static/launch-assemble.svg" style="max-width:100%;"></a>

## Data access

The data underlying the Fragile Families Challenge, which we are using in this collaboration, is sensitive and requires registration to access.

If you are already authorized to access the data, you can look over [Data Documentation](#data-documentation) below.

### Apply for access and registration

You must apply to Princeton's Office of Population Research (OPR) for access to the Fragile Families Challenge dataset.

> :envelope: [Follow instructions here to apply for access](https://docs.google.com/document/d/18uHdSS5NFNKmbYZvulsSkpgiA2NV9APmYQMgsQfI1_k/edit?usp=sharing)

**The Fragile Families Challenge dataset contains sensitive information. You should keep this dataset secure and protect the privacy of the individuals, and abide by the data access agreement which requires you not to share your copy of the dataset.**

You must register with us to join the collaboration, once you have been granted access to the data from Princeton OPR (or if you had already had access to the data from prior research). (This is step 7 in the instructions above, so don't repeat it if you already filled out the form.)

> :raised_hand: [Register here!](https://forms.gle/8MDLdZTftySqvn4e8)

### Authentication

Your AWS access key ID/secret will be automatically detected from standard locations (such as [environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables) or [credentials files](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file)).

If you are working in a notebook without access to other methods of configuration (such as using Assemblé) you can do the following in a code cell:

```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your access key id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your secret access key'
```

Alternatively, if you are working locally, you can create a new AWS profile in `~/.aws/credentials`:

```
[bff]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```

Then you can use this profile when you are developing features for this project, by exporting the environment variable `AWS_PROFILE=bff` (or using the `os.environ` approach similar to above).

## Data documentation

The full challenge dataset contains a "background" table of 4,242 rows (one per child in the training set) and 12,942 columns.

### Train split

The "train" split contains 2,121 rows (half of the background set) and 7 additional columns:

* `challengeID`: A unique numeric identifier for each child.
* Six outcome variables (each variable name links to a blog post about that variable)
  1. Continuous variables: `grit`, `gpa`, `materialHardship`
  2. Binary variables: `eviction`, `layoff`, `jobTraining`

These six outcome variables are the outcomes that we are trying to predict.

> :bulb: For the purpose of validating feature contributions, we will focus on
> the [`materialHardship`](https://www.fragilefamilieschallenge.org/material-hardship/) prediction problem. However, we want our feature
> definitions to be useful for all six prediction problems.

You can load the train split as follows:

```python
from ballet import b
X_df, y_df = b.api.load_data()
```

### Leaderboard and test splits

The other half of the rows are reserved for the "leaderboard" and "test" splits. We will use the leaderboard split to validate feature contributions. We will not look at the test split until the end of the collaboration.

If you'd like, you can load the full "background" dataset which includes the rows from the train, leaderboard, and test splits combined, but excluding the target columns.

```python
from fragile_families.load_data import load_background
background_df = load_background()
```

### Background variables

(This section is adapted from [here](https://www.fragilefamilieschallenge.org/apply/))

To use the data, it may be useful to know something about what each variable (column) represents. (See also the [full documentation](http://www.fragilefamilies.princeton.edu/documentation).)

#### Waves and child ages

The background variables were collected in 5 waves.

* Wave 1: Collected in the hospital at the child's birth.
* Wave 2: Collected at approximately child age 1
* Wave 3: Collected at approximately child age 3
* Wave 4: Collected at approximately child age 5
* Wave 5: Collected at approximately child age 9

Note that wave numbers are not the same as child ages. The variable names and survey documentation are organized by wave number.

#### Variable naming conventions

Predictor variables are identified by a *prefix* and a *question number*. Prefixes the survey in which a question was collected. This is useful because the documentation is organized by survey. For instance the variable `m1a4` refers to the `m`other interview in wave `1`, question `a4`.

1. The prefix c in front of any variable indicates variables constructed from other responses. For instance, `cm4b_age` is `c`onstructed from the `m`other wave `4` interview, and captures the child's age (`b`aby's `a`ge).
1. `m1`, `m2`, `m3`, `m4`, `m5`: Questions asked of the child's `m`other in wave `1` through wave `5`.
1. `f1`, `f2`, `f3`, `f4`, `f5`: Questions asked of the child's `f`ather in wave `1` through wave `5`
1. `hv3`, `hv4`, `hv5`: Questions asked in the `h`ome `v`isit in waves `3`, `4`, and `5`.
1. `p5`: Questions asked of the `p`rimary caregiver in wave `5`.
1. `k5`: Questions asked of the child (`k`id) in wave `5`
1. `ffcc`: Questions asked in various `c`hild `c`are provider surveys in wave 3
1. `kind`: Questions asked of the `kind`ergarten teacher in wave 4
1. `t5`: Questions asked of the `t`eacher in wave `5`.
1. `n5`: Questions asked of the `n`on-parental caregiver in wave `5`

### Full codebook

We expose the full [machine-readable codebook](https://www.fragilefamilieschallenge.org/machine-readable-fragile-families-codebook/) which you can use during feature development.

```python
from fragile_families.load_data import load_codebook
codebook_df = load_codebook()
```

### Metadata search

We also wrap the ffmetadata API for our own use in feature development. The metadata API returns more detailed metadata than is available in the codebook. See [here](https://github.com/fragilefamilieschallenge/metadata_app/blob/master/README.md#filter-specification) for details on the filter operations and see [here](http://metadata.fragilefamilies.princeton.edu/about) for an explanation of the resulting metadata.

```python
import fragile_families.analysis.metadata as metadata
metadata.info('m1a4')
metadata.search({'name': 'label', 'op': 'like', 'val': '%school%'})
# can use metadata.searchinfo to combine the two methods
```

#### Metadata search changes

The metadata search shows results from the most up-to-date metadata available. In some cases, this reflects changes since the 2017 challenge, so variables that appear in metadata search may not appear in the dataset and vice-versa.

For example, `kind_a2` was renamed to `t4a2`. If you use `metadata.info('kind_a2')` you will get an error. Instead, you would have to look at the `old_name` attribute:

```python
metadata.searchinfo({'name': 'old_name', 'op': 'eq', 'val': 'kind_a2'})
```

## Feature validation

In this project, feature contributions are [validated](https://ballet.github.io/ballet/contributor_guide.html#understanding-validation-results) to ensure that they are positively contributing to our shared feature engineering pipeline. One part of this validation is called "feature acceptance" validation, that is, does the performance of our ML pipeline improve when the new feature is added? We run the feature through two feature accepters: the [`MutualInformationAccepter`](https://ballet.github.io/ballet/api/ballet.validation.feature_acceptance.validator.html#ballet.validation.feature_acceptance.validator.MutualInformationAccepter) and the [`VarianceThresholdAccepter`](https://ballet.github.io/ballet/api/ballet.validation.feature_acceptance.validator.html#ballet.validation.feature_acceptance.validator.VarianceThresholdAccepter). Based on the parameters set in our [ballet.yml](ballet.yml#L17) configuration file, a feature definition is accepted if it meets two criteria:

- the variance of each its feature column values is greater than a threshold (set to 0.05), i.e. `Var(z_i) > 0.05 ∀ z_i ∈ z` where `z_i` are columns of feature `z`.
- the mutual information of the feature values with the target on the held out leaderboard dataset split is greater than a threshold (set to 0.001), i.e. `I(z ; y) > 0.001`.

## Discussion and help

Want to chat about the project, compare ideas, or debug features with other collaborators? Join either of our two chat rooms:

* Slack: [![slack](https://img.shields.io/badge/Slack-Join%20Channel-36C5F0?logo=slack)](https://join.slack.com/share/zt-r6u5ohtf-V40eunINB56ZD2Qd3b1L1w) (forgive me if the invite link is ever temporarily expired)
* Gitter: [![Join the chat at https://gitter.im/ballet-project/fragile-families](https://badges.gitter.im/ballet-project/fragile-families.svg)](https://gitter.im/ballet-project/fragile-families?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

If you think a question might have been answered before, check out the [Ballet FAQ](https://ballet.github.io/ballet/faq.html).

If you think you found a bug with Ballet, please [open an issue](https://github.com/ballet/ballet/issues/new) and mention that you are working on the ballet-fragile-families project.

## Additional resources

* [FF Data and Documentation](https://fragilefamilies.princeton.edu/documentation)
* [FF metadata homepage](http://metadata.fragilefamilies.princeton.edu/)
    * To see detailed metadata for a variable, can use the variables endpoint in your browser, like so (just replace the variable name):

      ```
      http://metadata.fragilefamilies.princeton.edu/variables/cf1cohm
      ```
* [metadata_app](https://github.com/fragilefamilieschallenge/metadata_app/blob/master/README.md)
* [ffmetadata-py](https://github.com/fragilefamilieschallenge/ffmetadata-py)
* [Machine readable codebook](https://www.fragilefamilieschallenge.org/machine-readable-fragile-families-codebook/)
* [Data dictionary in Excel](https://github.com/aarshayj/FragileFamiliesChallenge/blob/master/FFC_Data_Dictionary.xlsx)
* [Missing data in the challenge](https://www.fragilefamilieschallenge.org/missing-data/)
* Outcomes
    1. [Job training](https://www.fragilefamilieschallenge.org/job-training/)
    1. [Layoff](https://www.fragilefamilieschallenge.org/layoff/)
    1. [Eviction](https://www.fragilefamilieschallenge.org/eviction/)
    1. [Material hardship](https://www.fragilefamilieschallenge.org/material-hardship/)
    1. [Grit](https://www.fragilefamilieschallenge.org/grit/)
    1. [GPA](https://www.fragilefamilieschallenge.org/gpa/)
