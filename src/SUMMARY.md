# 인공지능 기초 (Fundamentals of Artificial Intelligence)

[서문](./intro.md)

# 시작하기 전에

- [왜 지금 AI인가?](./00-intro/01-why-ai.md)
- [AI, ML, DL의 차이](./00-intro/02-ai-ml-dl.md)
- [환경 설정](./00-intro/03-setup.md)
- [질문하는 법](./00-intro/04-how-to-ask.md)

# 1장. 파이썬 기초

- [첫 만남: print와 변수](./01-python-basics/01-hello.md)
- [자료형: 숫자, 문자, 참/거짓](./01-python-basics/02-data-types.md)
- [연산자: 더하기부터 비교까지](./01-python-basics/03-operators.md)
- [자료구조: 데이터를 담는 그릇](./01-python-basics/04-data-structure.md)
- [제어문: 조건과 반복](./01-python-basics/05-control-flow.md)
- [함수: 재사용 가능한 부품](./01-python-basics/06-functions.md)
- [실전 예제: 계산기 만들기](./01-python-basics/07-project.md)

# 2장. 머신러닝 — 분류

- [분류란 무엇인가](./02-ml-classification/01-what-is-classification.md)
- [데이터를 만나다 (MNIST)](./02-ml-classification/02-meet-data.md)
- [데이터 준비: 분할과 정규화](./02-ml-classification/03-data-preparation.md)
- [첫 모델 학습: SVM](./02-ml-classification/04-svm-intro.md)
  - [(보강) SVM 깊게 — 마진과 커널](./02-ml-classification/04a-svm-deep.md)
- [평가 지표: 정확도만으론 부족해요](./02-ml-classification/05-evaluation-metrics.md)
  - [(보강) 손실 함수 깊게](./02-ml-classification/05a-loss-function.md)
- [여러 모델 비교하기](./02-ml-classification/06-multiple-models.md)
- [하이퍼파라미터 튜닝](./02-ml-classification/07-hyperparameter-tuning.md)
- [현업 체크리스트](./02-ml-classification/08-production-checklist.md)
  - [(보강) 클래스 불균형 다루기](./02-ml-classification/08a-class-imbalance.md)

# 3장. 머신러닝 — 회귀

- [회귀란 무엇인가](./03-ml-regression/01-what-is-regression.md)
- [데이터를 만나다 (Boston)](./03-ml-regression/02-meet-data.md)
- [선형 회귀: 가장 단순하지만 강력한](./03-ml-regression/03-linear-regression.md)
- [회귀 평가 지표](./03-ml-regression/04-evaluation-metrics.md)
- [정규화: Lasso, Ridge, ElasticNet](./03-ml-regression/05-regularization.md)
- [트리 기반 회귀](./03-ml-regression/06-tree-regression.md)
- [현업 체크리스트](./03-ml-regression/07-production-checklist.md)

# 4장. 딥러닝 — 분류

- [신경망이란 무엇인가](./04-dl-classification/01-neural-networks.md)
- [PyTorch 시작하기](./04-dl-classification/02-pytorch-intro.md)
- [첫 신경망 만들기](./04-dl-classification/03-first-neural-network.md)
- [학습 과정 들여다보기](./04-dl-classification/04-training-process.md)
  - [(보강) 경사하강법 깊게](./04-dl-classification/04a-gradient-descent.md)
- [더 깊게: 활성화/손실/옵티마이저](./04-dl-classification/05-deep-dive.md)
  - [(보강) Batch Normalization 깊게](./04-dl-classification/05a-batch-norm.md)
- [GPU 활용](./04-dl-classification/06-gpu.md)
- [현업 체크리스트](./04-dl-classification/07-production-checklist.md)

# 5장. 딥러닝 — 회귀

- [회귀를 신경망으로](./05-dl-regression/01-neural-regression.md)
- [과적합이라는 적](./05-dl-regression/02-overfitting.md)
  - [(보강) 데이터 증강](./05-dl-regression/02a-data-augmentation.md)
- [무기 1: Early Stopping](./05-dl-regression/03-early-stopping.md)
- [무기 2: Dropout](./05-dl-regression/04-dropout.md)
- [무기 3: Weight Decay (L2)](./05-dl-regression/05-weight-decay.md)
- [세 무기 다 같이 쓰기](./05-dl-regression/06-combining-techniques.md)
- [현업 체크리스트](./05-dl-regression/07-production-checklist.md)

# 6장. 프로덕션으로 가는 길

- [모델 ≠ 서비스](./06-production/01-model-vs-service.md)
- [모델 패키징과 저장](./06-production/02-model-packaging.md)
- [FastAPI로 모델 서빙](./06-production/03-fastapi-serving.md)
- [Docker로 패키징](./06-production/04-docker.md)
- [모니터링과 데이터 드리프트](./06-production/05-monitoring.md)
- [MLOps의 큰 그림](./06-production/06-mlops.md)

# 부록

- [수학이 왜 필요한가](./appendix/math/00-why-math.md)
- [중학수학 복습](./appendix/math/01-middle-school-math.md)
  - [(심화) 중학수학 심화](./appendix/math/01a-middle-school-advanced.md)
- [고등수학 핵심](./appendix/math/02-high-school-math.md)
  - [(심화) 고등수학 심화](./appendix/math/02a-high-school-advanced.md)
- [미분 입문](./appendix/math/03-calculus-intro.md)
  - [(심화) 적분과 다변수](./appendix/math/03a-calculus-advanced.md)
- [선형대수 입문](./appendix/math/04-linear-algebra-intro.md)
  - [(심화) 선형대수 심화](./appendix/math/04a-linear-algebra-advanced.md)
- [확률통계 입문](./appendix/math/05-probability-stats-intro.md)
  - [(심화) 확률통계 심화](./appendix/math/05a-probability-stats-advanced.md)
- [ML에 적용하기](./appendix/math/06-ml-application.md)
- [(심화) 최적화 심화](./appendix/math/07-optimization-advanced.md)

---

- [신경망 가족 한눈에](./appendix/neural-networks-overview.md)
- [용어 사전](./appendix/glossary.md)
- [자주 묻는 질문](./appendix/faq.md)
- [정규화 총정리](./appendix/regularization-summary.md)
- [더 나아가기](./appendix/further-reading.md)
