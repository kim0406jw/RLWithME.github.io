---
sort: 2
---



# Trust Region Optimal Policy



# 0. Abstract

본 논문에서는 반복적 절차(Iterative Procedure)를 통한 정책 향상 알고리즘을 소개하고 있습니다. TRPO라고 불리우는 이 알고리즘은 이론적으로 정책향상을 보장하는 알고리즘을 실용적으로 적용 가능하게 근사한 것입니다. 이 알고리즘은 신경망과 같이 비선형 정책을 최적화하는데 효과적인 Natural Policy Gradient 방법과 비슷합니다. TRPO는 robotic swimming, hopping 등과 같은 다양한 Task에서 좋은 성능을 보이며 연속적인 행동 공간 제어와 관련해서 Policy Gradient의 가능성을 보여준 알고리즘입니다. 또한 TRPO는 연속적인 행동 공간 뿐만 아니라 이산적인 행동 공간 환경에서도 사용이 가능합니다.

TRPO는 Trust Region Policy Optimization의 약자로, 정책에 의해 행동해야 할 행동이 확률적으로 표현되는 ''확률론적 정책''의 향상을 보장하기 위해 Trust Region이라는 개념을 도입하였기에 이러한 이름이 붙여졌습니다. TRPO는 on-policy 알고리즘이기에 local-optima에 빠질 위험이 존재합니다. 그러나 local-optima에 빠질 지언정 이론적으로 정책 향상이 보장되는 장점을 가지고 있습니다. TRPO의 또다른 장점은 타 알고리즘에 비해 비교적 하이퍼-파라미터의 수가 적어 다양한 환경에서의 튜닝에 의한 노력이 크게 준다는 점입니다. 이는 TRPO가 상당히 일반화된 알고리즘임을 말하기도 합니다.





# 1. Introduction

최적 정책을 찾는 방법은 크게 3가지 카테고리로 묶을 수 있습니다.

1. **정책 반복 방법 (Policy Iteration Methods)**
   - 현재 정책 상에서 가치 함수를 평가하고, 이를 바탕으로 정책을 향상시키는 것을 반복하는 방법 
2. **정책 경사도 방법 (Policy Gradient Methods)**
   - 샘플 궤적(Sample Trajectories)하에 얻은 보상의 합의 기댓값의 기울기를 추정량으로 하여 정책을 학습하는 방법
   - 본 논문에서는 이후 정책 경사도 방법과 정책 반복 방법과의 연결점을 이야기하고 있습니다.

3. **Derivative-free 최적화 방법** 
   - Cross-Entropy Method(CEM)이나 Covariance Matrix Adaptation(CMA) Method와 같이 보상의 합을 black box 함수로두고 정책을 향상시키는 방법



 본 논문의 TRPO는 **정책 경사도 방법**을 이용한 알고리즘입니다.  정책 경사도 방법은 경사도를 이용하여 보상의 합(object function)을 증가시키는 방향으로 정책을 향상시키는데, first-order 경사도를 이용한다면, 곡선이 있는 영역에서 부정확 할 수 밖에 없습니다. 예를 들어 경사도 정책에서의 step-size(경사도를 바탕으로 얼마나 많이 갱신할 지의 지표) 가 크다면, 과장된 정책 향상의 방향을 그대로 믿고 정책 업데이트를 할 수도 있게 됩니다. 이렇게 되면 정책 향상이 올바르게 수렴하지 않을 가능성이 있습니다. 그렇다고 step-size를 너무 작게하면 학습의 속도가 너무 느려진다는 단점이 있습니다. 

앞으로 자세히 살펴보겠지만 TRPO는 이론적인 알고리즘을 실용적으로 사용하기 위해 근사한 알고리즘입니다. 이 근사된 알고리즘이 잘 통하기 위해서는 정책 갱신의 step-size가 충분이 작아야 합니다. 그렇다면 이 step-size가 얼마나 작아야할까요? 무턱대고 step-size를 매우 줄이면 학습 효율이 매우 떨어질겁니다. 

요약하자면 본 논문은,

1. 먼저 정책 향상이 보장되는 이론적인 알고리즘을 제시합니다.

2. 이 이론적인 알고리즘을 실용적으로 적용하기 위해 근사합니다.

3. 이렇게 근사된 알고리즘을 통해 정책을 최적정책으로 수렴시키고 싶습니다.

4. 그러나 갱신을 할 때 step-size가 너무 크면 정책이 잘 수렴하지 않습니다.

   -  step-size가 너무 크면 이론적인 알고리즘을 실용적인 알고리즘으로 근사할 수 없기 때문입니다.

   -  따라서 step-size의 크기에 대한 영향을 분석하고, 이에 따라 크기를 제한해야 합니다. 

5. 따라서 본 논문에서는 TRPO로 불리는 step-size의 크기를 제한하면서, 정책 향상이 보장된 근사된 알고리즘을 제안합니다.






# 2. Preliminaries

1. **할인이 포함된 MDP는 다음과 같이 정의 됩니다.**

 <div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104200857878.png" alt="image-20211104200857878"  /> </div>



2. **할인된 누적 보상의 기댓값(Expected Discounted Reward)** 

   현재 정책이![image-20211104201439937](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201439937.png) 이고, 초기 상태를 ![image-20211104201451829](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201451829.png)라고 한다면 이 정책을 따랐을 때 예상되는 보상의 합은 다음의 수식과 같습니다.

     <div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201505234.png" alt="image-20211104201505234" style="zoom:100%;" /> </div>

   

3. **행동가치 함수(state-action value function)** ![image-20211104201743415](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201743415.png), **가치 함수(value function)**![image-20211104201759527](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201759527.png)**, 그리고 Advantage function** ![image-20211104201831502](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104201831502.png)은 다음과 같이 표현할 수 있습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202044528.png" alt="image-20211104202044528" style="zoom:100%;" /></div>



4. **서로 다른 두 정책하에 얻어지는 리턴(Return)값의 관계**

   서로 다른 두 정책하에 얻어지는 리턴의 기댓값은 다음과 같은 관계가 있습니다.

   <div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202223223.png" alt="image-20211104202223223" style="zoom:100%;" /></div>

   표기법 ![image-20211104202317684](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202317684.png)은![image-20211104202334112](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202334112.png)로 샘플된 행동들을 의미합니다.

   

   위 식을 풀어 말하면, 

   ***"A라는 정책 하에서 얻어지는 리턴의 기댓값은,*** 

   1) ***B라는 정책 하에서 얻어지는 리턴의 기댓값과,***

   2) ***A 정책에서 얻어지는 상태-행동 trajectory에 대한 정책 B하의 Advantage 값을 할인해서 더한 값의 평균***

   ***을 더한 값이다."***

   라고 말할 수 있겠습니다.



식(1)의 증명은 다음과 같습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202448006.png" alt="image-20211104202448006" style="zoom:100%;" /></div>





5. **(unnormalized) discounted visitation frequencies**

   discounted visitation frequency란 한 상태를 방문할 할인된 확률의 합으로 정의 됩니다. 즉, ![image-20211104202608516](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202608516.png)를 discounted visitation frequency라고 한다면 다음과 같습니다.

   <div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202529289.png" alt="image-20211104202529289" style="zoom:100%;" /></div>

   

   여기서![image-20211104202627661](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202627661.png)이고, 행동들은 정책![image-20211104202645094](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202645094.png)를 따라 선택됩니다.





이제 식 (1)을 visitation frequency를 이용하여 다음과 같이 전개해나갈 수 있습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202710531.png" alt="image-20211104202710531" style="zoom:100%;" /></div>



이렇게 얻은 식(2)는 다음과 같은 의미를 내포하고 있습니다.

***"만약 모든 상태 s 에 대해서 advantage의 기댓값이 0 이상이라면, ![image-20211104202940902](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202940902.png)로의 정책 갱신은 policy performance ![image-20211104202958471](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202958471.png)를 증가시킴이 보장된다."***

즉, <img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202830851.png" alt="image-20211104202830851" style="zoom: 80%;" /> 라면, policy performance ![image-20211104203010016](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203010016.png)를 증가시킴이 보장된다고 말할 수 있습니다. 다시 말해 , <img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202830851.png" alt="image-20211104202830851" style="zoom: 80%;" />를 만족하는 정책![image-20211104203045052](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203045052.png)를 찾으면 ![image-20211104203021463](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203021463.png)를 증가시킬 수 있다는 것입니다.  즉, <img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104202830851.png" alt="image-20211104202830851" style="zoom: 80%;" />를 만족하는![image-20211104203053983](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203053983.png)로 기존 정책을 갱신해야한다는 갱신의  방향성을 알 수 있습니다.

그러한 정책 중 만약 ![image-20211104203100895](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203100895.png)를 ![image-20211104203128012](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203128012.png)로 정한다면 항상 ![image-20211104203143581](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203143581.png)를 만족할 것이고, 따라서 정책향상이 항상 보장될 것입니다. 이는 일반적인 deterministic policy로 ![image-20211104203155434](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203155434.png)를 사용하는 policy iteration방법과 매우 유사합니다. 이 유사성이 본 논문의 저자들이 말하는 policy iteration method와 policy gradient method의 연결이 될 수있습니다. 그러나 우리가 추정과 근사의 과정속에 발생하는 에러들로 인해 가끔 ![image-20211104203212157](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203212157.png) 이 되는 상태가 생길 수 있으며, 이는 보통 불가피합니다.



이제 다시 식

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203231019.png" alt="image-20211104203231019" style="zoom:100%;" /></div>

을 보면 ![image-20211104203303821](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203303821.png) 부분이 상당히 복잡합니다. 왜냐하면 보통 이전 정책으로부터 샘플링을 하고 이 샘플들을 바탕으로  정책경사도를 구하는데, ![image-20211104203310842](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203310842.png)은 갱신할 정책의 visitation frequency이기 때문입니다. 따라서 ![image-20211104203316229](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203316229.png)대신 이전 정책 ![image-20211104203326003](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203326003.png)을 사용하여 퍼포먼스를 계산하는 근사 식을 사용하는 것이 편리합니다. 따라서 본 논문에서는 다음과 같은 local approximation을 사용합니다. 직관적으로 매우 짧은 step-size라면, ![image-20211104203316229](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203316229.png)와 ![image-20211104203337028](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203337028.png)에서의 변화량은 크지 않을 것이며 따라서 근사식이 가능할 것이라고 생각할 수 있습니다.  

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203401337.png" alt="image-20211104203401337" style="zoom:100%;" /></div>

식(3)을 보면,![image-20211104203457905](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203457905.png)는 갱신된 정책![image-20211104203337028](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203337028.png) 대신 샘플링이 가능한 이전 정책 ![image-20211104203337028](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203337028.png))을 사용했습니다. 다시 말해 정책의 변화로 인한 state visitation density의 변화를 무시하였습니다. 이렇게 할 수 있는 이유는 다음과 같습니다. policy로 parameterized policy ![image-20211104203546170](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203546170.png)를 사용하고 ![image-20211104203559302](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203559302.png)가 미분 가능하다면, ![image-20211104203457905](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203457905.png)는 ![image-20211104203713648](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203713648.png)과 일차 근사가 일치하기 때문입니다. 이는 다음이 성립한다는 말과 같습니다. (Kakade & Langfold, 2002)

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203747588.png" alt="image-20211104203747588" style="zoom:100%;" /></div>

아래는 https://rll.berkeley.edu/deeprlcourse/docs/lec5.pdf에서 발췌한 식 (4)에 대한 설명이 있는 슬라이드 입니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203842911.png" alt="image-20211104203842911" style="zoom:100%;" /></div>

<center>[source: https://rll.berkeley.edu/deeprlcourse/docs/lec5.pdf]</center>



식 (4)는 ![image-20211104203916898](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203916898.png)을 증가시키는 ![image-20211104205605930](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205605930.png)로의 정책 갱신의 step-size를 충분히 작게 하면 ![image-20211104203713648](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203713648.png)역시 증가함을 뜻합니니다. 다시 말해 ![image-20211104203957966](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203957966.png)의 변화가 매우 작은 step동안은  ![image-20211104203929878](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203929878.png)와 ![image-20211104203713648](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104203713648.png)의 증가 폭이 같습니다.  이는 step-size를 충분히 작게하면 근사가 성립하며 정책 향상이 보장됨을 의미합니다. 그러나 식(4)는 이 근사가 성립하기 위해 step-size를 얼마나 작게해야하는지에 대한 가이드라인은 제공하고 있지 않습니다. 



6. **Conservative Policy Iteration**

   KaKade & Langford는 이러한 문제를 해결하기 위해 conservative policy iteration 이라는 정책 향상 방법을 제안하였습니다. 보수적인 정책 반복이라는 뜻을 가진 이 방법은 정책을 업데이트 할 때,  새로운 정책을![image-20211104204051109](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204051109.png) 으로 바로 사용하는 것이 아니라, 기존 정책과 ![image-20211104204105454](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204105454.png)의 적절한 비율로 나타내는 방법입니다.  따라서 정책의 갱신을 보수적으로(알파값이라는 비율을 도입하면서 step-size를 줄이면서)하는 것입니다. 즉, 업데이트될 새로운 정책을 다음처럼 표현할 수 있습니다. 

   <div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204118938.png" alt="image-20211104204118938" style="zoom:100%;" /></div>

   이런 식으로 정책을 갱신해 나간다면, step이 작기 때문에 상대적으로 안정적으로 수렴할 수 있게 됩니다.

   알파가 0이면 새로운 정책은 이전 정책과 같아지게 되고(성능의 갱신은 없다.), 알파가 1일 때, 성능 향상이 보장되려면 ![image-20211104204150476](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204150476.png)는 모든 상태에서 더 좋은 행동을 선택해야합니다.

   Kakade & Langford는 이렇게 새로운 정책을 현재 정책과 ![image-20211104204159756](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204159756.png)의 mixture로 표현한다면, 새로운 정책의 퍼포먼스는 다음과 같은 lower bound를 가짐을 보였습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204214951.png" alt="image-20211104204214951" style="zoom:100%;" /></div>



​		식(6)의 의미는 식(5) 로부터 얻은 실제 새로운 정책의 성능은 그 새로운 정책을 L (surrogate object) 이라는 함수에 대입한 값과 비교하였을 때, 적어도 ![image-20211104204240059](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204240059.png)보다 성능이 감소하지 않음을 보장해줍니다.



이제 새로운 정책의 퍼포먼스의 lower bound를 구하였으므로, lower bound를 최대화시켜 퍼포먼스를 최대화할 수 있게 되었습니다. 





# 3. Monotonic Improvement Guarantee for General Stochastic Polices

식(5) 에 등장하는 mixture policy는 일반적이지 않고 따라서 거의 쓰이지않는 비실용적인 방법입니다. 따라서 식(5)를 좀 더 일반적인 확률론적 정책에 관해 표현해야합니다. 본 논문에서는  ![image-20211104204302257](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204302257.png)를  old 정책 ![image-20211104204336589](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204336589.png)와 new 정책![image-20211104204351555](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204351555.png)분 포간의 거리의 지표 중 하나인 Total Variation Divergence를 사용하여 나타내어도 식 (6)이 성립함을 증명하였습니다. 조금 더 자세하게 ![image-20211104204310112](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204310112.png)를 풀어보면, 두 정책 분포의 각 상태마다 Total Variation Divergence 계산하였을 때, 가장 큰  Total Variation Divergence로  ![image-20211104204316064](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204316064.png)를사용합니다. 즉, ![image-20211104204321993](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204321993.png)로 다음을 사용합니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204404019.png" alt="image-20211104204404019" style="zoom:100%;" /></div>

![image-20211104204310112](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204310112.png)를 이렇게 대체하고, 식 (6)을 재정리하면 최종적으로 다음과 같은 정리가 완성됩니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204530411.png" alt="image-20211104204530411" style="zoom:100%;" /></div>

Theorem 1.로 인해 우리는 이제 일반적인 확률론적 정책에 사용이 가능하며, 정책 갱신에 대한 lower-bound를 얻게 되었습니다. 본 논문에서는 식(8)을 두 가지 방법으로 증명하였는데, 첫 번째 방법은 KaKade & Langford의 결과를 조금 더 확장하여 증명하는 방법이고, 두 번째 방법은 섭동이론(Perturbation Theory)로 증명하는 방법입니다. 두 방법 모두 본 논문의 Appendix에서 잘 설명하고 있습니다. 

다음으로 Total Variation Divergence는 KL Divergence보다 작기 때문에 lower-bound를 Total Variation Divergence 대신 KL Divergence로 표현할 수 있습니다. 즉, 아래의 식이 성립합니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204554347.png" alt="image-20211104204554347" style="zoom:100%;" /></div>

이제 이 lower-bound를 최대화 한다는 전략을 사용하여, 퍼포먼스의 향상을 보장하는 정책을 구하는 알고리즘을 봅시다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204631637.png" alt="image-20211104204631637" style="zoom:100%;" /></div>

알고리즘은 정책을 lower-bound를 최대화 시키는 정책으로 계속 수렴할 때 까지 반복적으로 갱신하게 됩니다.이러한 알고리즘은 minorization-maximization (MM) 알고리즘의 한 종류입니다. MM알고리즘에서 사용하는 용어로 ![image-20211104204703265](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204703265.png)는![image-20211104204715316](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204715316.png)를 minorize하는(그리고 ![image-20211104204715316](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204715316.png)과![image-20211104204734629](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204734629.png)에서는 같고) surrogate function을 의미합니다.  위 알고리즘을 시각적으로 표현하면 아래 그림과 같습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204825843.png" alt="image-20211104204825843" style="zoom:100%;" /></div>

<center>[source: https://www.youtube.com/watch?v=CKaN5PgkSBc&t=232s]</center>





# 4. Optimization of Parameterized Polices

이전 장에서 언급한 알고리즘은 이론적으로는 훌륭하지만 실제로 적용하기가 어렵습니다. 따라서 이번 장은 유한한 샘플 수와 임의의 파라미터화(finite sample counts and arbitrary parameterizations)에 대해서 작동하는 실용적인 알고리즘을 고안할 것입니다.  또한 이제부터 계속 벡터 ![image-20211104204921542](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204921542.png)하에  parameterized policy ![image-20211104204932033](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204932033.png)에 대해서 생각해 볼 것이기에 표기법에 있어서 다음과 같은 축약 표기법을 사용하겠습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205001278.png" alt="image-20211104205001278" style="zoom:100%;" /></div>

이전 장에서 실제 objective인![image-20211104204715316](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104204715316.png)를 증가시키기 위해 다음의 lower-bound를 최대화 시키는 전략을 사용하였습니다. 

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205049000.png" alt="image-20211104205049000" style="zoom:100%;" /></div>

실제로 위 최적화 문제를 그대로 풀어 새로운 정책을 찾는다면, 정책의 변화는 매우 작게 됩니다. 즉 정책 변화의 step-size가 매우 작아질 수 있습니다. step-size가 매우 작다면 수렴하는데 걸리는 시간과 샘플들이 많이 필요하기에  우리는 C를 어느정도 크게 업데이트를 하여 step-size를 키우고 싶습니다.  아래 그림은 C의 크기에 따른 step-size의 크기를 대략적으로 나타낸 것입니다. C가 클수록 ![image-20211104205117018](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205117018.png)가 최대가 되는 지점의 정책이 더 기존 정책에 비해 많이 변하는 것을 확인할 수 있습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205133359.png" alt="image-20211104205133359" style="zoom:100%;" /></div>

<center>[source: https://www.youtube.com/watch?v=CKaN5PgkSBc&t=232s]</center>



실용적으로 좀 더 큰 step-size를 가지게 할 수 있는(C를 키울 수 있는) 한 방법은 KL Divergence를 새로운 정책과 이전 정책의 제약식으로 표현하는 것 입니다. 이를 이용하여 식 (9)를 아래와 같은 최적화 문제로 변환할 수 있습니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205224732.png" alt="image-20211104205224732" style="zoom:100%;" /></div>

식(11)을 보면 KL Divergence의 값인 이전 정책과 새로운 정책의 거리를 'trust region'이라고 불리우는 크기로 제약을 겁니다. trust region은 학습을 할 때 수렴의 방향성을 벗어나지 않는 크기라고 생각하실 수 있습니다. 즉, trust region이 너무 커서 새로운 정책과 기존 정책의 KL Divergence가 커지는 것을 용인한다면, 최적 정책으로의 수렴은 보장할 수 없게됩니다.

한편 본 논문에서는 여기서 또 한번 근사를 진행을 합니다. 왜냐하면 식 (11)에서는 모든 state에 대해 KL Divergence의 최댓값을 찾아야하기 때문입니다. 이 작업을 간단하게 하기 위해 KL divergence의 최댓값을 평균값으로  heuristic하게 근사합니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205256656.png" alt="image-20211104205256656" style="zoom:100%;" /></div>

따라서 식 (11)은 최종적으로 다음과 같은 최적화 문제를 푸는것으로 바뀝니다.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205328578.png" alt="image-20211104205328578" style="zoom:100%;" /></div>



결국 TRPO는 식(12)를 푸는 알고리즘이라고 볼 수 있습니다.

# 5. Sample-Based Estimation of the Objective and Constraint

지금까지 파라미터로 근사된(예를 들어 신경망 혹은 선형 근사) 정책의 퍼포먼스의 기댓값을 최대화하기 위해 식(12)와 같은 제약 최적화 문제를 구했습니다. 이제 (12)를 토대로 최적 정책을 찾기만 하면 됩니다. 그런데 문제가 있습니다. 우리가 오라클과 같은 신적인 존재가 아니기에 샘플들의 정확한 평균 및 분포를 알 수가 없습니다. 따라서 에이전트와 환경의 상호작용을 통해 얻은 샘플을 가지고 평균 및 분산을 추정할 수 밖에 없는 추정의 문제가 생깁니다. 이 추정을 어떻게 하느냐에 따라 추정량과 그 추정량의 분산에 편향이 있을 수도 없을 수도 있습니다. 

이번 장에서는 '몬테 카를로' 를 통해 식(12)의 목적 함수(L, objective function)와 제약 함수(D, constraint function)를 어떻게 추정할지에 대한 내용을 담고있습니다.

가장 먼저 ![image-20211104205355036](https://raw.githubusercontent.com/RLWithME/RLWithME.github.io/develop/images/TRPO/image-20211104205355036.png)를 살펴보겠습니다.
