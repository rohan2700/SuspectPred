Abstract- This research paper proposes a fast prediction system for criminal suspects using ensemble learning algorithms. The system aims to forecast criminals efficiently by analyzing diverse crime dataset properties, including geographical location, crime types, historical crime rates, and socio-economic factors. The system utilizes Python, MySQL, and Flask for programming, data storage, and web application development. The algorithms implemented include ‘Logistic Regression, Linear Regression, Random Forest, Decision Tree, Extra trees classifier, and k-Nearest Neighbors (kNN)’ for predicting criminal activities. The feasibility study encompasses economic, technical, and social considerations, ensuring alignment with financial, technical, and societal parameters.
The proposed system effectively combines multiple classifiers, leading to improved prediction performance in identifying potential criminal suspects. The ensemble learning approach contributes to more effective criminal activity prevention plans and resource allocation. The system's high accuracy and potential to improve law enforcement decision-making make it a valuable tool for modern policing. However, ethical considerations and societal impact must be addressed, ensuring that the system respects civil liberties and privacy rights.
Keywords: Prediction, Criminal Suspects, Ensemble Learning Algorithms, Crime Types, Historical Crime Rates, Socio-economic Factors, ‘Logistic Regression’, ‘Linear Regression’, ‘Random Forest’, ‘Decision Tree’, ‘k-Nearest Neighbors’.

I.	INTRODUCTION

The fast prediction of criminal suspects is a critical area of research in the broad area of law enforcement and criminal justice. The primary goal of this project is to develop a system that can efficiently forecast potential criminal suspects by analyzing diverse crime datasets, including geographical location, crime types, historical crime rates, and socio-economic factors. The system employs a unique set of classifier algorithms, such as Random Forest, k Neighbors, Logistic Regression, Linear Regression, and Decision Tree, to suggest potential suspects. This predictive module enables swift identification of suspects, contributing to more effective and powerful criminal activity prevention strategies and resource allocation. 

The project follows a systematic approach, involving planning, development, testing, and deployment phases. The planning phase includes identifying stakeholders, defining project scope, conducting a feasibility study, establishing requirements, and creating a project plan. The development phase consists of designing architecture, data collections and preprocessing, algorithm development, software development, and integration and testing. The testing phase comprises unit-testing, integration-testing, system-testing, and user acceptance-testing (UAT). The deployment phase covers deployment planning, installation, training and documentation, monitoring and support, updates and enhancements, and data management.

The algorithms implemented in this paper include ‘Logistic Regression, Linear Regression, Random Forest, Decision Tree, and k-Nearest Neighbors (k-NN)’. Logistic Regression is a statistical method commonly used in ML for binary classification problems, and it can be applied to predict criminal activities according to crime dataset. In the area of criminal prediction, the algo aims to model the relationship in between a collection of independent variables (features extracted from the crime dataset) and the likelihood of a criminal event occurring (binary outcome – crime or no crime).

This idea to train and test models to ensure highest accuracy and precision has led to the implementation of multiple algorithms and compare them which might come in handy, these models are as follows:

Linear Regression: Linear-Regression is the most used method of predictive analysis. It uses linear relationships between a dependent variable (target) and a few independent variables (predictors) to predict the future of the target. The prediction is acted on the assumption that the relationship between the target and the predictors is dependent or causal.

Logistic Regression: Logistic Regression is a stat method commonly used in ML for binary classification problems, and it can be applied to predict criminal activities according to a crime dataset. In the context of criminal prediction, the algorithm aims to model the relationship between a set of independent variables (features extracted from the crime dataset) and the likelihood of a criminal event occurring (binary outcome – crime or no crime).

Random Forest Classifier: Random-Forest is a famous ensemble learning method that adds multiple decision trees to make more precise and robust predictions. In the context of crime prediction, the algorithm can be trained on a dataset containing various features such as geographical location, time of day, historical crime rates, socio-economic factors, and other relevant variables.

Decision Tree: Decision Tree is a prominent ML algorithm used for criminal prediction tasks on crime datasets. This classifier works by recursively dividing and partitioning the dataset into subsets based on the features that best separate instances of different classes. Each partition is represented by a tree node, and the process continues until a definite stopping criterion is met. In the context of criminal prediction, Decision Tree analyzes various features such as location, time, demographics, and historical criminal records to develope a predictive model.

k-Nearest Neighbors (k-NN): k-Nearest Neighbors is an ML algorithm employed for identification tasks, particularly in predicting whether a certain incident might lead to criminal activities. The k-NN belongs to the family of supervised learning algorithms and is based on the principle of proximity. It classifies data points by considering the majority class of their k-nearest neighbors in the feature space.

Extra Trees Classifier: The Extra Trees Classifier is a method that belongs to the Random Forest family. It is a variation of the Random Forest algorithm where many decision trees are created from the training-dataset. However, in the Extra Trees Classifier, the decision trees are constructed by random splits, and the best split is chosen from random features. This randomness in selecting splits helps to reduce overfitting and increase diversity among the trees, leading to improved generalization performance. Additionally, the Extra Trees Classifier introduces extra randomness by selecting random thresholds for each feature, further enhancing the model's robustness against noise and outliers.

The feasibility of this system is analyzed in three primary considerations: Economic Feasibility, Technical Feasibility, and Social Feasibility. Economic Feasibility involves evaluating the financial viability of the project, combining cost estimates for development, implementation, and maintenance. It also entails assessing potential returns on investment and determining if the benefits outweigh the costs over time. Technical Feasibility examines whether the technology required for the system is available, reliable, and scalable. It involves analyzing the capabilities of existing hardware, software, and data infrastructure to help the system's requirements effectively.

Social Feasibility considers the societal impact and acceptance of the predictive tool in law enforcement. It involves evaluating public perception, ethical considerations, and potential implications for civil liberties and privacy rights. Additionally, stakeholder engagement and community input are essential to gauge acceptance and address concerns.

The fast prediction of criminal suspects is a crucial area of research around law enforcement and criminal justice. By analyzing diverse crime dataset properties the system aims to forecast criminals efficiently. The ensemble learning approach effectively combines multiple classifiers, leading to improved prediction performance. The system's high accuracy and potential to improve law enforcement decision-making make it a valuable tool for modern policing. However, ethical considerations and societal impact must be addressed, ensuring that the system respects civil liberties and privacy rights.

II.	MOTIVATION 

Understanding the reason and motivation behind researching "Fast Prediction of Criminal Suspects" delves into the heart of societal concerns and the pursuit of justice. At its core, this topic emerges from a pressing need to enhance public safety while upholding the principles of fairness and human rights.

Firstly, swift identification of potential criminal suspects is crucial for crime prevention and mitigation. Time is often of the essence in law enforcement, particularly in cases of imminent threats or ongoing criminal activities. By developing efficient prediction methods, law enforcement agencies can swiftly allocate resources to investigate and apprehend suspects, potentially averting crimes before they occur.

Furthermore, the societal cost of crime, both in terms of economic impact and human suffering, underscores the urgency of accurate and rapid suspect identification. Timely identification and intervention can help solve these effects, fostering safer and more secure environments for all members of society.

Moreover, the pursuit of justice demands that law enforcement approaches be guided by principles of fairness and equity. By leveraging advanced technologies for suspect prediction, there is an opportunity to enhance the objectivity and transparency of law enforcement practices. This can help solve biases and ensure that each suspect is treated fairly under the law, regardless of factors such as race, ethnicity, or socioeconomic status.

In essence, the motivation for researching fast prediction of criminal suspects transcends mere technological advancement. It speaks to our collective desire for safer communities, the pursuit of justice, and the protection of fundamental human rights. By addressing these pressing societal needs, research in this field has the potential to make tangible and positive impacts on the lives of individuals and communities worldwide.

III.	LITERATURE SURVEY

The fast prediction of criminal suspects is an essential area of research in the field of law enforcement and criminal justice. By analyzing diverse crime datasets, including geographical location, crime types, historical crime rates, and socio-economic factors, predictive systems aim to forecast criminals efficiently. This paper shows and talks about a comprehensive review of the literature on fast prediction of criminal suspects, focusing on the utilization of ensemble learning algorithms.
Several researchers have explored the potential of ML techniques in predicting criminal activities. Ali et al. 1 proposed a machine learning-based predictive policing approach for criminal identification, demonstrating the effectiveness of utilizing historical crime data to predict future criminal activities. Similarly, Bharti and Mishra focused on predicting the network and role of offenders, highlighting the importance of understanding the relationships between different offenders.
Almuhanna et al. utilized spatial data analysis to predict crime in neighborhoods of New York City. Their study emphasized the significance of incorporating geographical information in predictive models. Agarwal et al. explored the application of statistical models in crime prediction, demonstrating the chance of using regression-based techniques for predicting criminal activities.
More recent researches have continued to explore the use of ML algorithms in predicting criminal activities. Wadhwa et al. proposed a combinational approach combining tracking criminal investigations and suspect prediction, while Safat et al. conducted an empirical analysis of crime prediction and forecasting using ML and DL (Deep learning) techniques. Yadav et al. focused on crime pattern detection, analysis, and prediction, emphasizing the importance of understanding crime patterns for effective crime prevention.
In summary, the literature on fast prediction of criminal suspects highlights the potential of ensemble learning algorithms in predicting criminal activities. By analyzing diverse crime datasets, including geographical location, crime types, historical crime rates, and socio-economic factors, predictive systems aim to forecast criminals efficiently. These studies demonstrate the effectiveness of using ML techniques in predicting criminal activities, emphasizing the importance of incorporating geographical information, understanding crime patterns, and considering the relationships between different offenders.

IV.	PROPOSED METHODOLOGY 

The series of steps that are followed in this project are as follows:
1.	Data Collection: Collect the relevant dataset for the problem statement. In this case, the dataset is "criminal_train.csv"
2.	Data Preprocessing:
a.	Load the dataset into a pandas Data Frame.
b.	Perform data cleaning by removing missing values.
c.	Convert categorical variables to numerical variables using one-hot encoding or label encoding.
d.	Scale the data using a suitable scaler, like RobustScaler.
3.	Feature Selection:
a.	Calculate the correlation between each feature and the target variable.
b.	Select highly correlated features according to the correlation values.
c.	Choose the most relevant features for the model.
4.	Data Splitting: Split the complete dataset into two, training and testing sets using the train_test_split function from the sklearn library.
5.	Data Balancing: Use the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to balance the dataset, if required.
6.	Model Selection:
a.	Implement various ML algorithms, like ‘Logistic Regression, KNN, Decision Trees, Random Forest, and Neural Networks.’
b.	Train each model on the training dataset.
c.	Evaluate each model on the testing dataset using appropriate metrics, such as accuracy, precision, recall, and F1-score.
7.	Model Comparison: Compare the performance of each algorithm using the evaluation metrics and choose the most accurate model.
8.	Model implementation: Train the selected model and test it for any desired input.
DETAILED IMPLEMENTATION: 

SMOTE (Synthetic Minority Over-sampling Technique) is a function used to handle imbalanced datasets by oversampling the minority class. It generates synthetic samples of the minority class by selecting instances from the minority class and creating new instances based on the difference between the selected instance and its nearest neighbors from the same class. This helps to balance the dataset, which can improve the performance and perfection of ML algorithms that are not indifferent to class imbalance.
With Accuracy Score calculated as the No. of correct predictions divided by the total No. of assumptions made. We compare each and every model. Considering the Extra Trees Classifies,

V.	RESULT AND ANALYSIS WITH GRAPHS

The system involves preprocessing the dataset, selecting relevant features, splitting the data into training and testing sets, and training various ML models to predict criminal activity. The preprocessing steps include converting categorical variables to numerical variables, scaling the data, and handling imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique). The relevant features are selected based on their correlation with the target variable, and the data is split into training and testing sets using a 75-25 split as seen in Figure 1.

To evaluate the accuracy and performance of the models, we use various evaluation methods such as accuracy, confusion matrix, and classification report. The accuracy metric measures the percentage of correct predictions out of the total No. of predictions made. The confusion matrix shows the No. of true positives (+), true negatives (-), false positives (+), and false negatives (-). The classification report shows the precision, recall, and F1-score for each and every class in a multi-class classification problem.

To compare the models and choose the best one, we evaluate each model on the testing dataset using the evaluation metrics. The model with the highest and best accuracy, precision, recall, and F1-score were chosen as the preferred model. In this problem, we train and evaluate various models such as ‘Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Extra Trees classifier’. The best model is chosen according to the evaluation metrics, and the trained model is saved to a file using joblib.

This proposed method involves a systematic approach to preprocessing the data, selecting relevant features, and training and evaluating various ML models. The evaluation metrics are used to compare the models and choose the best one. The trained model is saved to a file, making it easy to use the model for predictions in the future.


Table 1 A precise comparison of all the models based on Python’s sklearn metrics

Model	Accuracy	Precision	Recall	F1-score

Logistic  0.871  0.99  0.87  0.93

K-Nearest Neighbors  0.935  0.96  0.97  0.97

Decision Tree  0.887  0.98  0.89  0.94

Random Forest  0.89  0.99  0.89  0.94

Extra Trees classifier  0.887  0.98  0.89  0.94

The ‘Logistic Regression’ model achieves an overall Accuracy of 0.87, with a high Precision of 0.99, indicating a low false positive rate, and a Recall of 0.87, signifying its ability to identify true positives. Its F1-score of 0.93 suggests a good balance between Precision and Recall.
The K-NN model exhibits superior performance with an Accuracy of 0.94, coupled with high Precision (0.96) and Recall (0.97), resulting in an impressive F1-score of 0.97, indicating robust predictive capabilities.
The ‘Decision Tree, Random Forest, and Extra Trees classifier’ models demonstrate comparable performances, with accuracies ranging from 0.887 to 0.89. These models exhibit high Precision (0.98-0.99) and Recall (0.89-0.89), resulting in F1-scores ranging from 0.94 to 0.94, indicating their effectiveness in predicting criminal suspects.
In summary, while all models demonstrate strong predictive abilities, the K-NN model stands out for its superior performance across all metrics, followed closely by Logistic Regression. These findings as shown in Table 1 suggest the potential of ML algorithms in fast prediction of criminal suspects, with K-Nearest Neighbors being a promising candidate for practical implementation in law enforcement settings.
Now, The KNN model is selected as it shows the highest promise, and it is further used to identify if a particular person is a criminal or not. It displays a binary answer, either a “Criminal” or “Non-criminal”.

VI.	CONCLUSIONS

In summary, the research about "Fast Prediction of Criminal Suspects: An Ensemble Learning Approach" has yielded valuable insights into the potential application of advanced machine learning techniques in bolstering law enforcement capabilities. By employing ensemble learning methodologies, including Random Forest and Extra Trees classifier, alongside Logistic Regression, K-Nearest Neighbors, and Decision Trees, this study has demonstrated the effectiveness of a multifaceted approach in rapidly and accurately predicting criminal suspects.

The ensemble learning framework exhibited considerable strengths, particularly in achieving high levels of Accuracy, Precision, Recall, and F1-score across the various models assessed. This thorough examination showcased the robustness and adaptability of ensemble techniques in addressing complex real-world scenarios, where individual models may exhibit varying performance under diverse conditions.

Moreover, the findings underscored the importance of leveraging a diverse range of machine learning algorithms to mitigate biases and enhance the fairness and transparency of predictive policing practices. By amalgamating multiple models, each with its own unique strengths and limitations, law enforcement agencies can harness the collective intelligence of ensemble learning to make well-informed decisions and allocate resources efficiently.

However, it is crucial to acknowledge the limitations and ethical considerations associated with predictive policing. The potential for algorithmic biases and the necessity for continual monitoring and refinement to ensure equitable outcomes remain critical areas of concern.

In essence, it represents a significant stride towards harnessing the power of machine learning for the betterment of society. By embracing innovation while upholding principles of fairness, accountability, and human rights, we can work towards a future where predictive policing serves as a valuable tool in the pursuit of justice and public safety.
