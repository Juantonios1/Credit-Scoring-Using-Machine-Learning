# LIBRARY

# Basic Function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistic
import scipy.stats as stats
import math

# Ignore Warning
import warnings

warnings.filterwarnings("ignore")


# DISTRIBUTION


# Histogram
def hist(data, features):
    """
    data: df dataset.
    features: numerical features list.
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


# Bar Plot
def bar(data, features, target=None):
    """
    data: df dataset.
    features: categorical features list.
    target: label (string).
    """
    plt.figure(figsize=(15, 15))
    number = 1

    for feature in features:
        if target != None:
            if feature == target:
                continue
            plot = plt.subplot(math.ceil(len(features) / 2), 2, number)
            ax = pd.crosstab(data[target], data[feature]).apply(
                lambda r: r / r.sum() * 100, axis=1
            )
            ax_1 = ax.plot.bar(figsize=(10, 10), stacked=True, rot=0, ax=plot)

            plt.legend(loc=0, bbox_to_anchor=(0.1, 1.0), title="Subject")

            plt.title(feature, fontsize=20)
            for rec in ax_1.patches:
                height = rec.get_height()
                ax_1.text(
                    rec.get_x() + rec.get_width() / 2,
                    rec.get_y() + height / 2,
                    "{:.0f}%".format(height),
                    ha="center",
                    va="bottom",
                )
        else:
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


# Normal Test
def normal(data, features):
    """
    data: df dataset.
    features: categorical features list.
    """
    for feature in features:
        info, pval = stats.normaltest(data[feature])
        if pval > 0.05:
            print(feature, ": Normal distribution")
        else:
            print(feature, ": Not normal distribution")


# OUTLIER


# Boxplot
def box(data, features, hue=None):
    """
    data: df dataset.
    features: categorical features list.
    hue: label target (string).
    """
    plt.figure(figsize=(15, 15))
    number = 1

    for feature in features:
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        if hue != None:
            sns.boxplot(data=data, x=hue, y=feature, hue=hue)
        else:
            sns.boxplot(data=data[feature])
        plt.title(feature, fontsize=20)
        number += 1
        plt.tight_layout(h_pad=4)

    plt.show()


# IQR, Uppper, & Lower
def out(data, features):
    """
    data: df dataset.
    features: categorical features list.
    """
    for feature in features:
        Q1 = np.percentile(data[feature], 25)
        Q3 = np.percentile(data[feature], 75)
        IQR = Q3 - Q1
        uq = Q3 + 1.5 * IQR
        lq = Q1 - 1.5 * IQR

        print(
            f"""
        {feature}
        IQR : {IQR}
        Upper IQR : {uq}
        Lower IQR : {lq}"""
        )


##CORRELATION


# Scatter
def scatter(data, features, label):
    """
    data: df dataset.
    features: parameter list.
    label: target (string).
    """
    plt.figure(figsize=(15, 15))
    number = 1
    for feature in features:
        if feature == label:
            continue
        plt.subplot(math.ceil(len(features) / 2), 2, number)
        plt.scatter(data[feature], data[label], alpha=0.5)
        number += 1
        plt.title(f"{feature} to {label}", fontsize=16)
        plt.xlabel(f"{feature}")
        plt.ylabel(f"{label}")
        plt.tight_layout(h_pad=1)


# Correlation
def corr(data, feature, type="spearman"):
    """
    data: df dataset (include label).
    features: parameter list.
    type: "spearman" / "pearson" (default=spearman)
    """
    mask = np.triu(np.ones_like(data[feature].corr()))
    sns.heatmap(data.corr(type), annot=True, mask=mask)
    plt.title("Correlation Value", fontsize=15)
    plt.show()


# Chi Squared
def chi2(data, features, label):
    """
    data: df dataset.
    features: parameter list.
    label: target (string).
    """
    for feature in features:
        if feature == label:
            continue
        chisqt = pd.crosstab(data[label], data[feature], margins=True)
        value = np.array([chisqt.iloc[0], chisqt.iloc[1]])
        info, pval, dof, exp = stats.chi2_contingency(value)

        alpha = 0.05

        if pval <= alpha:
            print(feature, ": Dependent (reject H0)")
        else:
            print(feature, ": Independent (H0 holds true)")


# IV
def iv(data, features, label, target, non_target, decimal=3):
    """
    data: df dataset.
    features: parameter list.
    label: target (string).
    target: y(1) "string".
    non_target: y(0) "string".
    decimal: number of decimal in output.
    """
    iv = []

    # WOE
    for feature in features:
        if feature == label:
            continue
        df_woe_iv = (
            pd.crosstab(data[feature], data[label], normalize="columns")
            .assign(woe=lambda dfx: np.log(dfx[target] / dfx[non_target]))
            .assign(iv=lambda dfx: np.sum(dfx["woe"] * (dfx[target] - dfx[non_target])))
        )

        iv.append(df_woe_iv["iv"][0])

    # IV
    features_name = features
    if label in features_name:
        features_name.remove(label)

    df_iv = (
        pd.DataFrame({"Features": features_name, "iv": iv})
        .set_index("Features")
        .sort_values(by="iv")
    )
    df_iv.plot(
        kind="barh",
        title="Information Value of Categorical Features",
        colormap="Accent",
    )
    for index, value in enumerate(list(round(df_iv["iv"], decimal))):
        plt.text((value), index, str(value))

    plt.legend(loc="lower right")
    plt.show()


# Target Porportion
def tar_pro(data, label):
    """
    data: df dataset.
    label: target (string)
    """
    plt.figure(figsize=(17, (100) / 20))

    plt.subplot(121)
    plt.pie(
        round(data[label].value_counts() / len(data) * 100, 2),
        labels=list(data[label].value_counts().index),
        autopct="%.2f%%",
        explode=(0, 0.1),
    )
    plt.axis("equal")
    plt.title("Target Imbalance Ratio", size=15)

    plt.subplot(122)
    ax = sns.countplot(data=data, x=label)
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
