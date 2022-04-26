import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# GLOABL VARIABLES
path = "D:\\EPL-Result-Prediction\\Code\\"
file = "EPL2019-2020-modified.csv"

class DataFrame:

    def __init__(self):
        self.df = pd.read_csv(path + file, delimiter=";")
        self.pointsBeforeMatch()
        self.goalsBeforeMatch()
        self.matchesDf()

    # --- methods by instantiation ---

    def pointsBeforeMatch(self):
        pointsLambda = lambda row: row["current points"] - 3 if row["result"] == "w" else (row["current points"] - 1 if row["result"] == "d" else row["current points"] - 0)
        self.df["current points"] = self.df.apply(pointsLambda, axis=1)
        self.df.rename(columns={"current points": "current points before match"}, inplace=True)

    def goalsBeforeMatch(self):
        goalsLambda = lambda row: row["current total goals scored"] - row["goals scored"]
        self.df["current total goals scored"] = self.df.apply(goalsLambda, axis=1)
        goalsLambda = lambda row: row["current total goals conceded"] - row["goals conceded"]
        self.df["current total goals conceded"] = self.df.apply(goalsLambda, axis=1)

    def matchesDf(self):
        self.createMatchIDs()
        for counter, matchId in enumerate(list(self.df["match id"].unique())):
            singleMatchDf = self.df[self.df["match id"] == matchId]

            features = ["matchday", "date", "weekday", "referee",
                            "team", "goals scored", "current points before match",
                            "current total goals scored", "current total goals conceded"]

            homeTeamDf = singleMatchDf[singleMatchDf["home or away"] == "h"]
            homeTeamDf = homeTeamDf[features]
            homeTeamDf = self.renameColumns(features, homeTeamDf, "home")

            awayTeamDf = singleMatchDf[singleMatchDf["home or away"] == "a"]
            awayTeamDf = awayTeamDf[features]
            awayTeamDf = self.renameColumns(features, awayTeamDf, "away")

            matchDf = pd.merge(homeTeamDf, awayTeamDf)
            matchDf = self.pairColumns(list(homeTeamDf.columns), list(awayTeamDf.columns), matchDf)
            try:
                self.matchesDf = pd.concat([self.matchesDf, matchDf])
            except:
                self.matchesDf = matchDf

    def createMatchIDs(self):
        matchIDlambda = lambda row: str(row["date"]) + ": " + str(row["referee"])
        self.df["match id"] = self.df.apply(matchIDlambda, axis=1)

    def renameColumns(self, columns, df, homeOrAway):
        for column in columns:
            if column not in ["matchday", "date", "weekday", "referee"]:
                if column == "team":
                    df.rename(columns={column: f"{homeOrAway} {column}"}, inplace=True)
                else:
                    df.rename(columns={column: f"{homeOrAway} team {column}"}, inplace=True)
        return df

    def pairColumns(self, columns1, columns2, df):
        columnsOrdered = []
        for (column1, column2) in zip(columns1, columns2):
            if column1 == column2:
                columnsOrdered.append(column1)
            else:
                columnsOrdered.extend((column1, column2))
        return df[columnsOrdered]

    # --- methods by call ---

    def teamDf(self, teamName):
        df = self.matchesDf[(self.matchesDf["home team"] == teamName) | (self.matchesDf["away team"] == teamName)]
        df = self.turnIntoBinary(df, teamName)
        df = self.result(df)
        df = self.determineDifferences(df)
        return df

    def turnIntoBinary(self, df, teamName):
        homeLambda = lambda col: int(1) if col == teamName else int(0)
        df["home team"] = df["home team"].apply(homeLambda)
        df = df.drop(columns=["away team"])
        return df

    def result(self, df):
        res = lambda row: "win" if (row["home team"] == 1 and row["home team goals scored"] > row["away team goals scored"]) or \
            (row["home team"] == 0 and row["away team goals scored"] > row["home team goals scored"]) else \
            ("draw" if row["home team goals scored"] == row["away team goals scored"] else "loss")
        df["result"] = df.apply(res, axis=1)
        return df

    def determineDifferences(self, df):

        columnsOfInterest = ["goals scored", "current points before match",
                             "current total goals scored", "current total goals conceded"]
        for column in columnsOfInterest:
            differencesLambda = lambda row: (row[f"home team {column}"] - row[f"away team {column}"]) / row[
                "matchday"] if row["home team"] == 1 else (row[f"away team {column}"] - row[f"home team {column}"]) / \
                                                          row["matchday"]
            df[f"{column} difference per matchday"] = df.apply(differencesLambda, axis=1)
            df = df.drop(columns=[f"home team {column}", f"away team {column}"])
        return df

class MachineLearning:

    def __init__(self, df):
        self.df = df
        self.splitData()
        self.kNearestNeighbors()
        self.SVCmethod()
        self.randomForest()

    def splitData(self):

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.df[
                    ["home team", "goals scored difference per matchday",
                     "current points before match difference per matchday",
                     "current total goals scored difference per matchday",
                     "current total goals conceded difference per matchday"]],
                    self.df["result"], train_size=0.6, test_size=0.4, random_state=1)

    def kNearestNeighbors(self):
        k=3
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(self.xtrain, self.ytrain)
        accuracy = classifier.score(self.xtest, self.ytest)
        print(f"{k}: {accuracy}")
        ypredict = classifier.predict(self.xtest)
        self.plotConfusion(ypredict)

    def SVCmethod(self):
        classifier = SVC(C =0.1)
        classifier.fit(self.xtrain, self.ytrain)
        #support_vectors = classifier.support_vectors_
        ypredict = classifier.predict(self.xtest)
        accuracy = classifier.score(self.xtest, self.ytest)
        print(accuracy)
        self.plotConfusion(ypredict)

    def randomForest(self):

        classifier = RandomForestClassifier(n_estimators=3, random_state=0)
        classifier.fit(self.xtrain, self.ytrain)
        ypredict = classifier.predict(self.xtest)
        accuracy = classifier.score(self.xtest, self.ytest)
        print(accuracy)
        self.plotConfusion(ypredict)

    def plotConfusion(self, ypredict):
        confn = confusion_matrix(self.ytest, ypredict, labels=["win", "draw", "loss"])
        ax = sns.heatmap(confn, annot=True, cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Actual Values")
        ax.xaxis.set_ticklabels(["win", "draw", "loss"])
        ax.yaxis.set_ticklabels(["win", "draw", "loss"])
        plt.show()

if __name__ == '__main__':

    dataframe = DataFrame()
    teamDf = dataframe.teamDf("Man City")
    teamMachineLearning = MachineLearning(teamDf)