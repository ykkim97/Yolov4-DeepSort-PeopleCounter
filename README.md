## 실시간 객체인식기술을 이용한 출입자 통계시스템 개발
# 1. 프로젝트의 목적 및 용도

 코로나 사태 여파로 인해 생활 속 거리두기가 강조되는 요즘, 관광지나 다중이용 시설 등의 이용객수 밀집도를 파악하는 피플 카운터(People Counter)가 주목을 받고 있음.  
 피플 카운터는 오프라인 공간의 사람 수와 동선을 측정하여 결과를 분석해 비즈니스에 활용하도록 돕는 시스템이며 마케팅이나 재난상황 등 많은 곳에 활용이 가능함.  
 딥러닝 실시간 객체인식기술을 활용하여 특정 공간의 유동인원, 밀집도 등을 파악할 수 있는 시스템을 개발.  
 엣지컴퓨터에서 분석된 출입자에 대한 통계 정보들을 웹 브라우저에서 조회할 수 있도록 개발.  
 시스템을 통해 분석된 통계정보 결과는 빅데이터 분석에 활용될 수 있으며 엣지 컴퓨팅 방식을 이용하여 서버 부하를 경감시킬 수 있음.  

# 2. 사용 기술 및 언어
 Yolov4 & DeepSort  
 Python3.7 , Javascript  

# 3. 개발환경
 OS : Ubuntu 18.04  
 엣지컴퓨터(Jetson AGX Xavier) : CUDA-10.2, Tensorflow-2.3,1+nv20.10  
 서버(Raspberry Pi4) : Flask, Mariadb-10.3.25  
# 4. 동작 방식

 - 수 천여장의 이미지데이터를 수집한 후 학습시킴.  
 - 카메라가 부착된 엣지컴퓨터가 피플카운터를 실행하여 사람을 계측하여 DB에 저장한다.  
 - 이후 DB에 저장된 데이터를 요청하여 웹브라우저에 출력.  

# 참고자료 첨부(설계 및 세부 내용 관련)
실시간 객체인식 기술을 이용한 출입자 통계시스템 개발.pptx  
실시간 객체인식기술을 이용한 출입자통계시스템-보고서.hwp  

# References

Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:

tensorflow-yolov4-tflite
Deep SORT Repository
