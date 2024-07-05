import numpy as np
import sklearn
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
# # ################################
# # #  Non Editable Region Ending  #
# # ################################
	
	X_train_final=my_map(X_train)
	
	#BEST RESULTS OBTAINED USING LOGISTIC REGRESSION
	from sklearn.linear_model import LogisticRegression
	C=125
	tol=0.0025
	penalty='l2'
	solver='liblinear'
	max_iter=500

	clf=LogisticRegression(C=C,tol=tol,penalty=penalty,solver=solver,max_iter=max_iter)
	clf.fit(X_train_final,y_train)
	w=clf.coef_
	b=clf.intercept_

	return w,b



################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	no_challenges=X.shape[0]
	for i in range(no_challenges):
		X[i]=1-2*X[i]
		for j in range(32):
			for k in range(j+1,32):
				X[i][j]=X[i][j]*X[i][k]
	feat=[]
	for i in range(no_challenges):
		temp_feat=[]
		for j in range(32):
			temp_feat.append(X[i][j])
		for j in range(32):
			for k in range(j+1,32):
				temp_feat.append(X[i][j]*X[i][k])
		feat.append(temp_feat)
	feat=np.array(feat)
	return feat
	# return np.cumprod( np.flip( 2 * X - 1 , axis = 1 ), axis = 1 )


X=np.array([[1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,1],[1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1]])
y=np.array([1,0])
w,b=my_fit(X,y)
print(w.shape)
print(w)
print(b.shape)
print(b)