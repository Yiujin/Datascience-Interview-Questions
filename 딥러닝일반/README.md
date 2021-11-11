## 딥러닝
## 딥러닝 일반
- 딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?
> 딥러닝은 레이어를 깊게 쌓는 방식으로 모델링   
> 딥러닝은 학습과정에서 feature를 직접 찾고, 머신러닝은 사람이 feature를 직접 지정해줘야함 

- 왜 갑자기 딥러닝이 부흥했을까요?
> 여러개 layer + ReLU activation funtion으로 비선형 문제를 해결할 수 있어서   
> GPU 및 자원의 발전으로 아주 큰 딥러닝을 학습시킬수 있는 충분한 자원이 생겨서

- 마지막으로 읽은 논문은 무엇인가요? 설명해주세요
> Transformer
> encoder, decoder 구조를 가지며, self-attention을 통해 단어간의 관계와 단어의 중요성을 파악하여 전체 문장에서의 맥락을 파악함 

- Cost Function과 Activation Function은 무엇인가요?
> Cost Function : Loss function. 신경망이 '최적의 가중치', 즉 최적의 parameter(매개변수)를 찾게 만드는 '지표'
> Activation Function : 이전 레이어들로부터 온 값들, 즉 입력 신호의 총합을 출력 신호로 변환하는 함수이다. activation이라는 단어 자체에서 알 수 있듯이, 입력 신호의 총합이 다음 레이어의 활성화를 일으킬지 말지를 결정한다.

> ref  
> https://mole-starseeker.tistory.com/38

- Tensorflow, Keras, PyTorch, Caffe, Mxnet 중 선호하는 프레임워크와 그 이유는 무엇인가요?
> Pytorch  
> 코드가 직관적이며 이해하기 쉬움. 
> 텐서플로우와 비교했을 때 가장 큰 차이점은 딥러닝 구현 패러다임이다. 텐서플로우 : Define and Run은 실행시점에 데이터만 바꿔도 되는 유연함이 장점이지만 비직관적이며 난이도가 높은 반면 파이토치: Define by Run 방식의 파이토치는 선언과 동시에 이뤄져 간결하고 난이도가 낮다.

  -> Define and Run : 1) 코드를 돌리는 환경인 세션 2) placeholder 선언 3) 계산 그래프 선언 4) 코드 실행

  -> Define by Run : 1) 선언과 동시에 데이터 삽입 및 실행

> ref  
> https://m.blog.naver.com/dsz08082/222122736953

- Data Normalization은 무엇이고 왜 필요한가요?
> 입력데이터의 범위를 0~1 사이로 조절하는것, 모델이 학습과정에서 scale이 큰 변수에 편향되는 것을 방지하여 학습을 원활하게 함

- 알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)
> Sigmoid : 0과 1사이로 만듬. 항상 1보다 작은 수를 return하므로 역전파시 gradient vanishing 문제를 일으킴.
> ReLU : sigmoid의 기울시 소실 문제를 해결하기 위한 활성화 함수. 0보다 큰 입력값는 해당 값그대로, 0보다 작은 입력값은 0으로 만듬 / 0보다 작은 입력값은 모두 0이 되어 해당 노드가 죽어 모델의 표현력이 제한되는 문제점 있음
> LeakyReLU : ReLU에서 0보다 작은 입력값에 약간의 기울기를 주어 표현력을 높임
> Tanh : -1과 1사이로 만듬

- 오버피팅일 경우 어떻게 대처해야 할까요?
> 데이터 부족인지, 데이터에 비해 너무 큰 모델을 쓴건지 원인을 생각해보고 데이터 증강 및 모델 크기를 작게 만든다.

- 하이퍼 파라미터는 무엇인가요?
> learning rate, batch size와 같이 학습과정에서 사람이 직접 정해줘야 하는 파라미터

- Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?
> 가중치 초기화는 학습에 사용되는 가중치들을 어떤 분포를 따르도록 초기화 하는것. 이를 통해 back propagation 때 가중치의 기울기값이 넓게 퍼지도록 하여 모델의 표현력을 높이고, 기울기가 0이 되는 gradient vanishing현상을 방지한다.
> He 초기화 : ReLU activation function을 쓸 때 사용하는 가중치 초기화 방법으로, 해당 레이어의 가중치를 2/이전 레이어의 노드갯수 의 제곱근값으로 초기화
> Xavier 초기화 : sigmoid activation function을 쓸 때 사용하는 가중치 초기화 방법으로, 해당 레이어의 가중치를 1/이전 레이어의 노드갯수 의 제곱근 값으로 초기화





- 볼츠만 머신은 무엇인가요?
- 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?
	- Non-Linearity라는 말의 의미와 그 필요성은?
	- ReLU로 어떻게 곡선 함수를 근사하나?
	- ReLU의 문제점은?
	- Bias는 왜 있는걸까?
- Gradient Descent에 대해서 쉽게 설명한다면?
	- 왜 꼭 Gradient를 써야 할까? 그 그래프에서 가로축과 세로축 각각은 무엇인가? 실제 상황에서는 그 그래프가 어떻게 그려질까?
	- GD 중에 때때로 Loss가 증가하는 이유는?
	- 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
	- Back Propagation에 대해서 쉽게 설명 한다면?
- Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
	- GD가 Local Minima 문제를 피하는 방법은?
	- 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
- Training 세트와 Test 세트를 분리하는 이유는?
	- Validation 세트가 따로 있는 이유는?
	- Test 세트가 오염되었다는 말의 뜻은?
	- Regularization이란 무엇인가?
- Batch Normalization의 효과는?
	- Dropout의 효과는?
	- BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
	- GAN에서 Generator 쪽에도 BN을 적용해도 될까?
- SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?
	- SGD에서 Stochastic의 의미는?
	- 미니배치를 작게 할때의 장단점은?
	- 모멘텀의 수식을 적어 본다면?
- 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
	- 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
	- Back Propagation은 몇줄인가?
	- CNN으로 바꾼다면 얼마나 추가될까?
- 간단한 MNIST 분류기를 TF, Keras, PyTorch 등으로 작성하는데 몇시간이 필요한가?
	- CNN이 아닌 MLP로 해도 잘 될까?
	- 마지막 레이어 부분에 대해서 설명 한다면?
	- 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
	- 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?
- 딥러닝할 때 GPU를 쓰면 좋은 이유는?
	- 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
	- GPU를 두개 다 쓰고 싶다. 방법은?
	- 학습시 필요한 GPU 메모리는 어떻게 계산하는가?
- TF, Keras, PyTorch 등을 사용할 때 디버깅 노하우는?
- 뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가? 