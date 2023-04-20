# Image_translation_program-feat.GoogleLens
이미지를 입력하면 원하는 언어로 추출하여 번역해주는 프로그램입니다.

![image](https://user-images.githubusercontent.com/101550112/233429998-001143bd-33de-42bd-bc0a-67666e24b108.png)

전체 프로그램 설명 URL : https://youtu.be/ZOzcWPnJ5WA
<br>
<br>
<br>

Project 폴더 - 전체적 코드 존

발표관련 폴 - 전체 프로그램 설명 URL의 발표 내용 제작에 사용된 자료


## 개요
이미지를 넣으면 OCR을 통해 글자를 인식해서 번역하여 이미지에 적용해주는 프로그램입니다.

- 이미지를 넣어서 탐지된 텍스트를 번역한 후 이미지에 적용.


- 탐지된 텍스트와 번역된 텍스트 확인 기능.


- 탐지된 텍스트의 threshold 확인 가능.

<br>

## 기술
구현 도구 : Visual Studio Code

구현 코드 : Python

<br>

## 실행 화면

### ※ 자세한 작동기전은 영상을 참고해주세요. ( https://youtu.be/ZOzcWPnJ5WA )

<br>

프로그램 실행시 다음과 같이 3개의 탭이 있습니다.


![image](https://user-images.githubusercontent.com/101550112/233297295-89ab3b2d-2568-4aa8-9467-4fccebda1e37.png)

1번 탭 Image Upload : 직접 이미지 파일을 올려서 쓰레기를 분류해 봅니다.

![image](https://user-images.githubusercontent.com/101550112/233296961-479fd597-ecfe-4112-b651-56a4dee99a1b.png)

2번 탭 Using WebCam : 내장되어있는 캠을 통해 이미지를 생성하고 쓰레기를 분류합니다.

![image](https://user-images.githubusercontent.com/101550112/233304292-7bfe2f15-8593-461c-b9ce-e0bbee21ea77.png)

3번 탭 Making Image : 원하는 쓰레기 이미지를 생성하여 해당 이미지를 갖고 해당 프로그램을 테스트 해봅니다.

![image](https://user-images.githubusercontent.com/101550112/233321363-f3b6eca3-1f10-4920-aa0e-e67a46f6a158.png)

<br>

각 탭마다 아래에는 다음과 같이 3개의 라디오 버튼이 있는데, 넣은 이미지의 쓰레기 종류를 파악하여 정보를 제공해 줍니다.

![image](https://user-images.githubusercontent.com/101550112/233424128-5581c757-c248-4cb8-b6d9-adafb255df20.png)

![image](https://user-images.githubusercontent.com/101550112/233424513-59de4535-2ee8-4174-974f-431e3e5fd31b.png)
<br>
