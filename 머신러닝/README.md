## 머신러닝
- Cross Validation은 무엇이고 어떻게 해야하나요?
> cross validation이란 데이터를 학습용 데이터와 검증용 데이터로 나누어 학습용 데이터로만 모델을 학습시키고, 한번도 학습시키지 않은 검증데이터에 대해 모델의 성능을 확인하는 과정.
Cross Validation의 방법으로는 Hold out, k-fold cross validation, leave one and out 등의 방법이 있다.   <br><br>
hold out : 전체 데이터를 8:2 또는 9:1의 비율로 학습 : 검증 데이터로 나눔. 장점 : 구현이 간단하고 쉬움 / 단점 : 검증 데이터만큼의 데이터를 학습에 사용하지 못하는 문제  
k-fold cross validation : hold out의 단점을 해결하기 위해 k번의 fold로 나누어 k 번 학습하고, 각 fold에서 train, valid data를 바꾸어가며 학습해서 모든 데이터를 학습에 사용하는 방법. 장점 : 모든 데이터를 학습에 사용 가능 / 단점 : 학습 시간이 오래 걸림  
leave one and out : 단 하나의 데이터만 검증용으로 사용하고, 나머지는 모두 학습용으로 사용. 장점 : 학습 데이터의 양이 줄어들지 않음 / 단점 : 검증에 대한 신뢰도가 떨어짐


- 회귀 / 분류시 알맞은 metric은 무엇일까요?
> 회귀 : MSE, RMSE, MAE, accuracy 등
분류 : accuracy, f1 score 등

- 알고 있는 metric에 대해 설명해주세요(ex. RMSE, MAE, recall, precision ...)
> MSE : 정답과 예측 데이터 간 차이의 제곱을 모두 더해 평균낸 값   
RMSE : MSE 의 양의 제곱근   
MAE : 정답과 예측 데이터 간 차이의 절대값을 모두 더해 평균낸 값   
accuracy : 정확도. confusion matrix에서 TP+FP / (TP+FP+TN+TP) 의 비율  
recall : 재현율. 실제로 참인 것중에 모델이 참이라고 예측한 비율.  confusion matrix에서 TP / (TP + FN) 의 비율  
precision : 정밀도. 모델이 참이라고 예측한 것중에 실제로 참인 비율. confusion matrix에서 TP / (TP + FP) 의 비율  
F1 score : recall 과 precision의 조화평균. 데이터 불균형이 심한 경우 정확도는 올바른 metric이 되지 않아 사용함.  

- 정규화를 왜 해야할까요? 정규화의 방법은 무엇이 있나요?
> 수치형 데이터 간의 scale이 맞지 않으면 모델이 학습과정에서 값이 더 크거나 작은 (극단적인) 데이터에 편향되게 되고, 정확한 결과를 얻을 수 없으므로 데이터의 scale을 맞추어 주는 정규화를 해야함.  <br><br>
min-max scaling:  데이터에서 최소값을 뺴고 최대값과 최소값의 차이로 나누어 데이터의 범위를 [0,1]로 만드는 방법.   
z-score scaling:  데이터를 표즌화(평균을 뺴고 표준편차로 나눔)하여 평균이 0이고 표준편차가 1인 분포로 만드는 것.   
robust scaling:  IQR(사분위값 기준 Q3-Q1의 값)의 값을 1로 만드는 정규화. 이상치가 많은 데이터에 적용하는 정규화 방법. 
>> IQR은 데이터의 중앙 50%를 의미한다. 사분편차는 (Q3-Q1)/2 값인데, 사분편차가 의미하는 것은 데이터의 분포가 퍼져있는 정도이다. 사분편자의 값이 크다면 데이터가 넓게 분포한다는 것이고(즉 scale이 크다는 것) 사분편차의 값이 작으면 데이터가 좁게 분포한다는 것이다. robust scaling은 사분편차의 두배인 IQR값을 1로 만들어줌으로서 데이터의 scale을 줄이는 방법. 

> ref  
https://drhongdatanote.tistory.com/30
https://wotres.tistory.com/entry/Robust-scaling-%ED%95%98%EB%8A%94%EB%B2%95-in-python


- Local Minima와 Global Minima에 대해 설명해주세요.
> 

- 차원의 저주에 대해 설명해주세요
- dimension reduction기법으로 보통 어떤 것들이 있나요?
- PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?
- LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?
- Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?
- 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?
- SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? 거기서 어떤 장점이 발생했나요?
- 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.
- Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.
- 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
- 머신러닝(machine)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
- 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?
- 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
- ROC 커브에 대해 설명해주실 수 있으신가요?
- 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?
- K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)
- L1, L2 정규화에 대해 설명해주세요
- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
- 앙상블 방법엔 어떤 것들이 있나요?
- SVM은 왜 좋을까요?
- feature vector란 무엇일까요?
- 좋은 모델의 정의는 무엇일까요?
- 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?
- 스팸 필터에 로지스틱 리그레션을 많이 사용하는 이유는 무엇일까요?
- OLS(ordinary least squre) regression의 공식은 무엇인가요?