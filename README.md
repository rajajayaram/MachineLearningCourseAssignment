# MachineLearningCourseAssignment
Coursera Learning Course Assignment

**1.Overview**

This analysis is the final report of the Peer Assessment project from Coursera’s
course Practical Machine Learning, as part of the Specialization in Data Science.
This analysis meant to be the basis for the course quiz and a prediction assignment
writeup.

The main goal of the project is to predict the manner in which 6 participants
performed some exercise as described below.
There is the “classe” variable in the training set.
The machine learning algorithm described here is applied to the 20 test cases
available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading

**2.Background Details**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible
to collect a large amount of data about personal activity relatively inexpensively.

These type of devices are part of the quantified self movement - a group of
enthusiasts who take measurements about themselves regularly to improve their
health, to find patterns in their behavior, or because they are tech geeks. One
thing that people regularly do is quantify how much of a particular activity
they do, but they rarely quantify how well they do it. 

In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX

**Overview of the Dataset**

The training data for this project are available here:

## [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) ##

The test data are available here:
## [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) ##

The data for this project come from http://groupware.les.inf.puc-rio.br/har.
A short description of the datasets content from the authors’ website:
"Six young health participants were asked to perform one set of 10 repetitions of
the Unilateral Dumbbell Biceps Curl in five different fashions: 
exactly accordingto the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other
4 classes correspond to common mistakes. Participants were supervised by an
experienced weight lifter to make sure the execution complied to the manner
they were supposed to simulate. The exercises were performed by six male
participants aged between 20-28 years, with little weight lifting experience. We
made sure that all participants could easily simulate the mistakes in a safe and
controlled manner by using a relatively light dumbbell (1.25kg)."
