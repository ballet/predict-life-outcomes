[![ballet](https://img.shields.io/static/v1?label=built%20with&message=ballet&color=FCDD35)](https://ballet.github.io)
<a href="https://mybinder.org/v2/gh/HDI-Project/ballet-fragile-families/master?urlpath=lab" target="_blank" rel="nofollow"><img src="https://ballet.github.io/ballet/_static/launch-assemble.svg" style="max-width:100%;"></a>

# Fragile Families Collaboration

This is a collaborative predictive modeling project built on the [ballet framework](https://ballet.github.io).

The Fragile Families Challenge (FFC) is a recent attempt to better connect to the social science research community to new tools in data science and machine learning. This challenge aimed to spur the development of predictive models for life outcomes from data collected as part of the Fragile Families and Child Wellbeing Study (FFCWS), which collects detailed longitudinal records on a set of disadvantaged children and their families. Organizers released anonymized and merged data on a set of 4,242 families with data collected from the birth of the child until age 9. Participants in the challenge were then tasked with predicting six life outcomes of the child or family at age 15: child grade point average, child grit, household eviction, household material hardship, primary caregiver layoff, and primary caregiver participation in job training. The FFC was run over a four month period in 2017 and received 160 submissions from social scientists, machine learning practitioners, students, and others.

In this project, we ask, by collaborating rather than competing, can we develop impactful solutions to the FFC? Participants in the FFC were competing against each other to produce the best performing models, at the expense of collaboration across teams.

## Join the collaboration

Are you interested in joining the collaboration?

- Read the [Ballet Contributor Guide](https://ballet.github.io/ballet/contributor_guide.html)
- Read the [Ballet Feature Engineering Guide](https://ballet.github.io/ballet/feature_engineering_guide.html)
- Browse the currently accepted features in the contributed features
    directory ([`src/fragile_families/features/contrib`](src/fragile_families/features/contrib))
- Launch an interactive Jupyter Lab session to hack on this repository:
    <a href="https://mybinder.org/v2/gh/HDI-Project/ballet-fragile-families/master?urlpath=lab" target="_blank" rel="nofollow" ><img src="https://ballet.github.io/ballet/_static/launch-assemble.svg" style="max-width:100%;"></a>

## Data access

The data underlying the Fragile Families Challenge, which we are using in this collaboration, is sensitive and requires registration to access. More details are upcoming about how to access this data.

If you are authorized to access the data, you can do so as follows:

```python
from ballet import b
X_df, y_df = b.api.load_data()
```

Your AWS access key ID/secret will be automatically detected from standard locations (such as [environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables) or [credentials files](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file)) by the data loading in order to provide you access to the data.

For example,

```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your access key id
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your secret access key'
```
