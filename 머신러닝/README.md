## 머신러닝
- Cross Validation은 무엇이고 어떻게 해야하나요?
> cross validation이란 데이터를 학습용 데이터와 검증용 데이터로 나누어 학습용 데이터로만 모델을 학습시키고, 한번도 학습시키지 않은 검증데이터에 대해 모델의 성능을 확인하는 과정.
Cross Validation의 방법으로는 Hold out, k-fold cross validation, leave one and out 등의 방법이 있다.   <br><br>
hold out : 전체 데이터를 8:2 또는 9:1의 비율로 학습 : 검증 데이터로 나눔. 장점 : 구현이 간단하고 쉬움 / 단점 : 검증 데이터만큼의 데이터를 학습에 사용하지 못하는 문제  
k-fold cross validation : hold out의 단점을 해결하기 위해 k번의 fold로 나누어 k 번 학습하고, 각 fold에서 train, valid data를 바꾸어가며 학습해서 모든 데이터를 학습에 사용하는 방법. 장점 : 모든 데이터를 학습에 사용 가능 / 단점 : 학습 시간이 오래 걸림  
leave one and out : 단 하나의 데이터만 검증용으로 사용하고, 나머지는 모두 학습용으로 사용. 장점 : 학습 데이터의 양이 줄어들지 않음 / 단점 : 검증에 대한 신뢰도가 떨어짐


- 회귀 / 분류시 알맞은 metric은 무엇일까요?
> 회귀 : MSE, RMSE, MAE, MAPE 등  
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
min-max scaling:  데이터에서 최소값을 빼고 최대값과 최소값의 차이로 나누어 데이터의 범위를 [0,1]로 만드는 방법.   
z-score scaling:  데이터를 표즌화(평균을 빼고 표준편차로 나눔)하여 평균이 0이고 표준편차가 1인 분포로 만드는 것.   
robust scaling:  IQR(사분위값 기준 Q3-Q1의 값)의 값을 1로 만드는 정규화. 이상치가 많은 데이터에 적용하는 정규화 방법. 
>> IQR은 데이터의 중앙 50%를 의미한다. 사분편차는 (Q3-Q1)/2 값인데, 사분편차가 의미하는 것은 데이터의 분포가 퍼져있는 정도이다. 사분편자의 값이 크다면 데이터가 넓게 분포한다는 것이고(즉 scale이 크다는 것) 사분편차의 값이 작으면 데이터가 좁게 분포한다는 것이다. robust scaling은 사분편차의 두배인 IQR값을 1로 만들어줌으로서 데이터의 scale을 줄이는 방법. 

> ref  
https://drhongdatanote.tistory.com/30
https://wotres.tistory.com/entry/Robust-scaling-%ED%95%98%EB%8A%94%EB%B2%95-in-python


- Local Minima와 Global Minima에 대해 설명해주세요.
> loss funtion이 최소화 되는 지점(최소값)을 찾는 것이 모델 학습의 목적인데, 이 지점을 Global Minima라고 한다.  
> 최소값이 아닌 극소값이 있을 수 있는데, 이 점을 local minima라고 한다. local minima가 아닌 global minima를 찾도록 도와주는 역할을 하는 것이 optimizer.


- 차원의 저주에 대해 설명해주세요
> 데이터의 수보다 차원이 커 모델 성능이 저하되는 현상. 차원이 증가할 수록 개별 차원내의 데이터수가 적어져 데이터를 표현하기 어렵기 때문에. 

> ref  
> https://datapedia.tistory.com/15


- dimension reduction기법으로 보통 어떤 것들이 있나요?
> 차원을 감소하기 위한 방법으로는 feature selection과 feature extraction이 있다. 
> feature selection은 말 그대로 모델 학습에 사용할 변수를 선택하는 기법으로, 결측값이 많은 변수 제거, 다중공선성이 높은 변수 중 하나 제거, 전진선택법/후진제거법/단계적 선택법 과 같이 모델의 성능 향상에 기여도가 낮은 변수 제거와 같은 방법이 있다.
<br>
feature extraction은 변수들의 차원을 줄이는 기법으로, PCA(주성분 분석), SVD, ICA, auto-encoder, LDA 등이 있다.  
PCA(Principal Component Analysis): 주성분분석. 여러 변수를 주성분이라는 서로 상관성이 높은 변수들의 선형결합으로 만들어 기존의 상관성이 높은 변수들을 축소하는 방법. 즉, 데이터가 정규분포를 따른다고 가정할 수 있을 때, 분포된 데이터들의 주성분를 찾을 때 데이터를 가장 잘 설명할 수 있는 축을 찾는 기법이다.  PCA에서 주성분은 이 축을 의미하며 분산(Variation)이 최대가 되는 축이 데이터를 가장 잘 설명하는 축이다. 알고리즘은 축소하고자 하는 데이터 간의 공분산행렬을 구하고 고유벡터를 추출해 PCA값을 산출.
<br><br>
ICA(Independent Component Analysis): 독립성분 분석. 데이터가 통계적으로 독립하여 정규분포를 따르지 않는다는 가정이 가능할 때, 독립성이 최대가 되는 방향으로 축을 얻는 방법이다. 즉, 주성분을 이용한다는 점에서 PCA와 비슷하지만 데이터를 가장 잘 설명하는 축을 찾는 PCA와 달리 가장 독립적인 축을 찾는다. 이에 독립성(Independence)이 최대가 되는 벡터를 찾는다.
<br><br>
SVD(Singular Value Decomposition): 차원 감소 방법 중 특이값 분해(Singular Value Decomposition, SVD)은 임의의 행렬을 세 행렬의 곱으로 분해하며 수식은 아래와 같다.
$X=USV^T$
(U,V: 직교행렬(Orthogonal matrix), 그 열벡터는 서로 직교한다.
S: 대각행렬(diagonal matrix), 대각성분 외에 모두 0인 행렬)
직교행렬 U는 어떤 공간의 축(기저)을 형성하며, 대각행렬 S의 대각성분에는 "특잇값(해당 축의 중요도)"이 큰 순서로 나열되어 있다.
특잇값이 작다면 중요도가 낮다는 뜻이므로 행렬 S에서 여분의 열벡터를 제거하여 원래의 행렬을 근사할 수 있다. 데이터가 linear한 상황에서 잘 동작한다. rmse를 목표로, latent vector를 중간에 두고 원래의 행렬을 복원
<br><br>
auto-encoder: 뉴럴네트워트를 이용한 데이터 축소 기법. input data와 output data가 같은 것으로 하여 인코더는 데이터를 낮은 차원으로 압축하고, 디코더가 다시 latent vector를 입력데이터와 같은 크기로 만들어서 원래의 데이터를 복원함. SVD와 동일한 아이디어지만 이 네트워크는 훨씬 더 복잡하게 응용이 가능하고, non-linear problem solving에서도 아주 좋은 성능을 보여준다.
<br><br> 
LDA(Linear Discriminant Analysis): 선형판별분석. pca보다 분류문제에 최적화 되어있는 차원축소 기법. lda는 분류할 수 있게 분별기준을 최대한 유지하며 축을 설정함. 지도적 방식으로 데이터의 분포를 학습하여 분리를 최적화하는 피처 부분공간을 찾은 뒤, 
학습된 결정 경계에 따라 데이터를 분류하는 것이 목표이다.
PCA의 경우는 변수간의 공분산 행렬을 이용하여 데이터를 대표하는 에이겐 쌍을 추출했다면
LDA의 기본적인 방법은 class들의 mean 값들의 차이는 최대화하는 행렬 A, 
class내의 variance는 최소화하는 행렬 B를 찾아내서 B의 역행렬 X A행렬에다가 eigendicomposition을 통해 
class 간의 mean은 최대화하고 class 내의 variance는 최소화하는 에이겐 쌍을 추출한다.

<br><br>
요인분석(Factor Analysis): 등간척도로 측정한 두 개 이상의 변수들에 잠재되어있는 공통인자를 찾아내는 기법. 

> ref  
> https://ichi.pro/ko/chawon-gamso-176985025504183  
https://engineer-mole.tistory.com/48
https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/04/06/pcasvdlsa/
https://velog.io/@guide333/%EB%B0%91%EC%8B%9C%EB%94%A5-2%EA%B6%8C-2.4-%ED%86%B5%EA%B3%84-%EA%B8%B0%EB%B0%98-%EA%B8%B0%EB%B2%95-%EA%B0%9C%EC%84%A0%ED%95%98%EA%B8%B0#243-svd%EC%97%90-%EC%9D%98%ED%95%9C-%EC%B0%A8%EC%9B%90-%EA%B0%90%EC%86%8C
https://yamalab.tistory.com/116
https://yamalab.tistory.com/41

- PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?
> 주성분 분석의 기본적인 개념은 차원이 큰 벡터에서 선형 독립하는 고유 벡터만을 남겨두고 차원 축소를 하게 됩니다.
이때 상관성이 높은 독립 변수들을 N개의 선형 조합으로 만들며 변수의 개수를 요약, 압축해 내는 기법입니다. 
그리고 이 압축된 각각의 독립 변수들은 선형 독립, 즉 직교하며 낮은 상관성을 보이게 됩니다.
가령 500차원의 벡터를 주성분 분석한다는 것은 각 차원의 분산을 최대로 갖는, 분포를 설명할 수 있는 대표축을 뽑는 과정이고, 
주성분 분석결과 나오는 매트릭스에서 PC1 은 각 칼럼에 대한 정보 설명력이 PC5~6에 비해 높습니다. 
이처럼 높은 주성분들만 선택하면서 정보 설명력이 낮은, 노이즈로 구성된 칼럼들은 배제하기 때문에
노이즈 제거 기법이라고 불리기도 합니다.

>ref   
https://huidea.tistory.com/126


- LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?
> Latent Sematic Analysis(잠재의미분석) : pca에서 확장된 차원 축소 기법, 지도학습에서 사용되며 입력데이터의 클래스를 최대한 분리할 수 있는 축을 찾는 기법
> Linear Discriminant Analysis(선형판별분석) : 
> Singular Value Decomposition(특이값분해) : 정사각행렬이 아닌 m*n 형태의 다양한 행렬을 분해하며, 
이때 분해되는 행렬은 두 개의 직교 행렬과 하나의 대각행렬이고 두 직교행렬에 담긴 벡터가 특이벡터.



- Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?
> 여러 State를 갖는 Chain 형태의 구조를 일컫는다. 무엇이 되었건 State가 존재하고, 각 State를 넘나드는 어떤 확률값이 존재하며, 다음 State는 현재 State 값에만 의존(Markov Property)한다면, 이는 모두 Markov Chain이다.


- 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?
> 1. 텍스트의 맞춤법을 교정 후 띄어쓰기 또는 형태소 기준으로 나누기
> 2. 나눈 데이터에서 같은 형태인 데이터의 갯수를 세어 통계내기
> 3. 해당 통계를 통해 어느 주제에 해당하는지 판단하기
> 다양한 주제에 대한 글에서 어느 단어가 얼마나 나오는지 통계값 정보를 알고 있어야 함
> 동음이의어의 경우 주제 판단에 영향을 끼칠 수 있을 듯


- SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? 거기서 어떤 장점이 발생했나요?
> support vector machine은 분류문제에 사용하는 방법으로 분류결정경계에서 가장 가꺼운 데이터들간의 거리인 마진을 최대화하는 초평면을 찾는 기법이다. (선형분류) 하지만 직선으로는 두 범주를 완벽히 분류하기 어려운 경우도 있는데, 이 경우 kernel trick을 사용하여 저차원의 데이터를 고차원으로 변환한 후 결정경계를 찾고, 비선형 경계면을 찾을 수 있다는 장점이 있다. 
https://ratsgo.github.io/machine%20learning/2017/05/30/SVM3/

- 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.
> 결과 도출을 위해 조건부 확률만 계산하면 되므로 매우 빠르고, 메모리를 많이 차지하지 않는다.
데이터가의 특징들이 서로 독립되어 있을 때 좋은 결과를 얻을 수 있다.
데이터의 양이 적더라도 학습이 용이하다.

> ref  
> https://yongwookha.github.io/MachineLearning/2021-01-29-interview-question

- Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.
> Association Rule : 데이터 내부의 상호관계 또는 종속관계를 찾아내는 분석 기법
> Support : 지지도. X와 Y 두 item이 얼마나 자주 발생하는지를 의미
> Confidence : 신뢰도. X가 발생했을 떄, Y도 포함되어 있는 비율
> Lift : 향상도. X가 발생하지 않았을 때의 Y 발생 비율과 X가 발생했을 때 Y 발생 비율의 대비. 숫자가 1과 같으면 서로 독릭적, 1보다 크면 양의 상관관계, 1보다 작으면 음의 상관관계.


- 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
> Gradient Descent : 현재 시점의 가중치에서 loss function 의 기울기를 계산하여 loss 가 작아지는 방향(극소값)으로 이동하며 최적화를 하는 방법.  
> 장점 : 모든 차원과 모든 공간에서 적용 가능  
> 단점 : 최솟값을 찾지 못하고 극소값에 빠질수 있음  
> Newton's Method : 방정식 f(x) = 0의 해를 근사적으로 찾을 때 사용되는 방법.
  현재 x값에서 접선을 그리고 접선이 x축과 만나는 지점으로 x를 이동시켜 가면서 점진적으로 해를 찾아가는 방법  
> 장점 : 초기값을 잘 주면 해를 금방 찾을 수 있음  
> 단점 : 해를 아예 찾지 못할 수도 있음

> ref  
> https://astralworld58.tistory.com/86

- 머신러닝(machine)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
> 머신러닝이 결국 통계를 기반으로 한 접근 방법이다. 다만 머신러닝은 학습 과정에서 사람이 아닌 모델이 최적화를 해 나간다. 

- 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?
> 비선형적인 문제 해결할 수 없음. 


- 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
> 여러개로 쌓은 레이어와 ReLU 활성화 함수를 사용하여 비선형함수를 근사하도록 한 것


- ROC 커브에 대해 설명해주실 수 있으신가요?
> 이진분류기의 성능을 판단하기 위하여 가능한 모든 threshold에 대한 True Positive rate과 False Psotive Rate을 그래프로 그린 것. 이때 threshold란 모델이 데이터의 정답을 예측하는 기준을 의미하며 모든 데이터에 대해 True라고 예측하면 threshold는 낮아진다. ROC curve는 커브 아래의 면적(AUC)이 넓을수록 모델의 성능이 좋다.  
> 

>ref  
https://angeloyeo.github.io/2020/08/05/ROC.html  
그리는 방법 :https://hsm-edu.tistory.com/1033


- 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?
> 


- K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)
> 군집의 중심점을 정할 때 군집에 속한 데이터의 평균값을 통해서 중심점을 업데이트하므로 이상치에 민감하다.   
> 이를 해결하기 위해 평균값 대신 중앙값을 사용하는 K-Median 방법이 있다.

- L1, L2 정규화에 대해 설명해주세요
> 모델의 과적합을 막기 위한 정규화 방법으로,  
> L1 정규화는 cost function에 `가중치의 절댓값`을 더하여 cost function값을 커지게 해 weight가 지나치게 커지는 것을 막는다. 불필요한 weight는 편미분시 0이 되는 효과가 있다. L1 정규화를 사용한 선형회귀가 Lasso.  
> L2 정규화는 codt function에 `가중치의 제곱한 값`을 더하여 weight 가 커지는 것을 막는다. 이를 wwight decay라고 한다. L2 정규화를 사용한 선형회귀가 Ridge.

> ref  
> https://light-tree.tistory.com/125

- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
> Gradient Boosting 알고리즘을 분산환경에서도 실행할 수 있도록 구현해놓은 라이브러리이다. Regression, Classification 문제를 모두 지원하며, 성능과 자원 효율이 좋아서, 인기 있게 사용되는 알고리즘이다.
여러개의 Decision Tree를 조합해서 사용하는 Boosting Ensemble 알고리즘. Boosting 은 이전 모델이 잘못예측한 값에 가중치를 주어 다음 모델이 더 잘 예측하도록 순차적으로 학습하는 방법. 
<br>
>장점  
유연한 Learning 시스템 - 여러 파라미터를 조절해가면서 최적의 Model을 만들 수 있음.  
Over fitting(과적합)을 방지할 수 있다.  
신경망에 비해 시각화가 쉽고, 이해하기보다 직관적이다.  
자원(CPU, 메모리)이 많으면 많을수록 빠르게 학습하고 예측할 수 있다.  
Cross validation을 지원한다.  
높은 성능을 나타낸다.   


> ref  
> 출처: https://bcho.tistory.com/1354 [조대협의 블로그]
> https://dining-developer.tistory.com/3


- 앙상블 방법엔 어떤 것들이 있나요?
> Bagging
여러 모델을 병렬적으로 학습하여 결과를 합하는 방법으로 분산을 줄이는 효과, Random Forest 

> Boosting
여러 모델을 순차적으로 학습하여 이전모델이 잘못예측한 것을 다음모델이 더 잘 예측하도록 만드는 방법으로 편향을 줄이는 효과, XGBoost, Gradient Boosting

- SVM은 왜 좋을까요?
> SVM은 데이터들을 선형 분리하며 최대 마진의 초평면을 찾는 크게 복잡하지 않은 구조이며, 커널 트릭을 이용해 차원을 늘리면서 비선형 데이터들에도 좋은 결과를 얻을 수 있다. 또한 이진 분류 뿐만 아니라 수치 예측에도 사용될 수 있다. Overfitting 경향이 낮으며 노이즈 데이터에도 크게 영향을 받지 않는다.

>ref  
https://excelsior-cjh.tistory.com/166

- feature vector란 무엇일까요?
> 입력 데이터, 가중치 등 머신러닝에 사용되는 변수들을 벡터형태로 만든 것


- 좋은 모델의 정의는 무엇일까요?
> 편향되지 않고 모델의 예측이 사회적으로 문제가 되지 않는 모델


- 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?
> 의사결정나무의 단점은 분산이 크다는 것이다. 50 개의 작은 의사결정 나무를 모아 앙상블하면 분산을 감소시키는 효과가 있다. 


- 스팸 필터에 로지스틱 리그레션을 많이 사용하는 이유는 무엇일까요?
> 성능이 좋고 계산 비용이 저렴해서 


- OLS(ordinary least squre) regression의 공식은 무엇인가요?
> 최소제곱법, 근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법  
> https://yongwookha.github.io/MachineLearning/2021-01-29-interview-question