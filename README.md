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

프로그램 시작하면 2개의 탭이있습니다. 
각 탭은 이미지를 컴퓨터에서 갖고올지, 웹캠을 통해 갖고올지 그 차이만 있을 뿐, 기능은 동일합니다. 

![image](https://user-images.githubusercontent.com/101550112/233433978-3d7c4f13-3c3e-4a34-ab68-a6fbacc84bc9.png)


프로그램에 사용할 이미지를 넣은뒤 Choice Detecting Language 란의 라디오 버튼과 Choice Transform Language 란의 라디오 버튼 지정하고 TransForm Image를 누르면 자동으로 Choice Detecting Language 의 언어를 파악하여 Choice Transform Language언어로 번역하여 이미지를 보여줍니다.

![image](https://user-images.githubusercontent.com/101550112/233434504-7e3edd79-d699-4cfc-ae2e-d811628332ee.png)

위 사진의 경우 여러 글자가 있으나 Choice Detecting Language 란의 ko ( 한국어 ) 만 디텍팅 하여 Choice Transform Language 란의 언어 ja ( 일본어 )로 변경된 것을 볼 수 있습니다.

![image](https://user-images.githubusercontent.com/101550112/233435688-e52a8beb-d567-4c52-92fa-e4378aa770b6.png)

웹 사이트 아래 부분에서는 디텍팅한 글자가 무엇인지, confidence score가 몇이 나왔는지 확인해 볼 수 있습니다.

<br>
