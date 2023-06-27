# Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os


# Numerical Distribution
def num_dist(features, data):
    """
    input:(features= list of numerical feature, data= dataframe of dataset)
    output: distribution graphic

    """
    plt.figure(figsize=(15, 13))
    number = 1

    for feature in features:
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        sns.kdeplot(data[feature], shade=True, alpha=0.5, linewidth=4.2)
        plt.title(feature, fontsize=22)
        number += 1
        plt.tight_layout(h_pad=4)

    plt.show()


# # Categorical Distribution

# In[13]:


def cat_dist(features, data):
    """
    input:(features= list of categorical feature, data= dataframe of dataset)
    output: distribution graphic

    """
    plt.figure(figsize=(15, 15))
    number = 1

    for feature in features:
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        ax = sns.countplot(data=data.fillna("NaN"), x=feature)
        plt.title(feature, fontsize=20)
        for p in ax.patches:
            ax.annotate(
                f"\n{p.get_height()}",
                (p.get_x() + 0.4, p.get_height()),
                ha="center",
                color="black",
                size=14,
            )
        number += 1

    plt.tight_layout(h_pad=2)


# ## Outlier Identification (Boxplot)

# In[4]:


def boxplot(features, data):
    plt.figure(figsize=(15, 15))
    number = 1

    for feature in features:
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        sns.boxplot(data=data[feature])
        plt.title(feature, fontsize=20)
        number += 1
        plt.tight_layout(h_pad=4)

    plt.show()


# ## WOE & IV

# In[14]:


def iv_funct(features, data, label, target, non_target):
    """
    input:(features= list of categorical feature, data= dataframe of dataset,
        label= target column ("string"), target= y(1) "string", non_target= y(0) "string")
    output: WOE graphic

    """
    iv = []

    # WOE
    for feature in features:
        df_woe_iv = (
            pd.crosstab(data[feature], data[label], normalize="columns")
            .assign(woe=lambda dfx: np.log(dfx[target] / dfx[non_target]))
            .assign(iv=lambda dfx: np.sum(dfx["woe"] * (dfx[target] - dfx[non_target])))
        )

        iv.append(df_woe_iv["iv"][0])

    # IV
    df_iv = (
        pd.DataFrame({"Features": features, "iv": iv})
        .set_index("Features")
        .sort_values(by="iv")
    )
    df_iv.plot(
        kind="barh",
        title="Information Value of Categorical Features",
        colormap="Accent",
    )
    for index, value in enumerate(list(round(df_iv["iv"], 3))):
        plt.text((value), index, str(value))

    plt.legend(loc="lower right")
    plt.show()


# ## Scatterplot

# In[ ]:


def scatter(data, features, label):
    """
    input:(dataset= dataframe, feature= list of parameter, label=target)
    output: scatter plot

    """
    plt.figure(figsize=(15, 15))
    number = 1
    for feature in features:
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        plt.scatter(data[feature], data[label], alpha=0.5)
        number += 1
        plt.title(f"{feature} to {label}", fontsize=16)
        plt.xlabel(f"{feature}")
        plt.ylabel(f"{label}")
        plt.tight_layout(h_pad=1)


# ## Correlation

# In[1]:


def corr(dataset, label, feature, corr_type="spearman"):
    """
    input:(dataset= dataframe, label=target, feature= list of parameter, corr_type= spearman/pearson (default=spearman))
    output: correlation image

    """
    mask = np.triu(np.ones_like(dataset[feature].corr()))
    sns.heatmap(dataset.drop([label], axis=1).corr(corr_type), annot=True, mask=mask)
    plt.title("Correlation Value", fontsize=15)
    plt.show()


# ## Label Proportion

# In[15]:


def lab_pro(dataset, label):
    """
    input:(dataset= dataframe of dataset, label= target column ("string"))
    output: WOE graphic

    """
    plt.figure(figsize=(17, (100) / 20))

    plt.subplot(121)
    plt.pie(
        round(dataset[label].value_counts() / len(dataset) * 100, 2),
        labels=list(dataset[label].value_counts().index),
        autopct="%.2f%%",
        explode=(0, 0.1),
    )
    plt.axis("equal")
    plt.title("Target Imbalance Ratio", size=15)

    plt.subplot(122)
    ax = sns.countplot(data=dataset, x=label)
    plt.title("Barplot Target Label", fontsize=15)
    for p in ax.patches:
        ax.annotate(
            f"\n{p.get_height()}",
            (p.get_x() + 0.4, p.get_height()),
            ha="center",
            va="top",
            color="white",
            size=12,
        )
