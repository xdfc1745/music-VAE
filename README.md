## music-VAE
Music genration을 위한 모델로, 문장과 달리 훨씬 긴 sequence를 생성하는 모델.  
sequence가 길어지면서 “posterior collapse(RNN 모델이 잠재 벡터를 무시하고 학습에 집중하는 현상)”문제가 발생할 수 있는데 이를 해결하기 위해서 Hierarchical recurrent decoder를 도입하였다.  
<img width="398" alt="image" src="https://github.com/xdfc1745/music-VAE/assets/39234312/290b6c37-e2ef-482d-93b0-ce03f1aa1092">  
위의 그림처럼 decoder 부분을 conductor와 Decoder 부분으로 나눠서 hierarchical 하게 구성되어 있다.  

입력 값 (midi data)을 encoder에서 Latent vector(잠재벡터)로 바꿔준다. 이 값을 다시 Hierarchical Decoder에 넣어주면 입력 값가 비슷하지만 새로운 midi data를 생성한다.

## install
이 코드를 사용하기 위해서 `magenta`를 설치해야한다.  
현재 m1 os 상에서는 설치가 안되는 것으로 보이나, 인텔 맥에서는 설치가 된다.  
```
conda create -n music-vae python=3.8.16

conda activate music-vae
pip install magenta
pip install note-seq
```
conda 환경을 구축 후, magenta를 설치했다. 하지만 conda를 사용하지 않고 python3.8.x의 환경에서 설치하면 된다.

## Preprocessing
midi파일을 학습하기 위해서는 먼저 벡터화를 해주어야 한다.  
magenta의 convert_directory를 이용해서 벡터 파일인 TFRecord 파일로 변경해주었다. 이는 구글의 Protocal Buffer 포맷으로 데이터를 직렬화하여 저장한 파일 형태이다.  
논문에서 4/4박자의 파일만을 사용한다고 해서 4/4박자가 아닌 파일은 제거해주었다.   

https://magenta.tensorflow.org/datasets/groove   
데이터는 위의 링크에서 goovae-v1.0.0-midionly 파일을 다운받은 후, 압축을 풀어주었다.  

```
python preprocessing.py \
--input_dir=groove \
--output_file=music.tfrecord \
--recursive \
--is_beat=True \
--beat=4-4
```

## Train model
위에서 TFRecord 파일로 바꾼 데이터를 이용하여 모델을 학습시킨다.  

```
python train.py \
--config=groovae_4bar \ 
--mode=train \
--run_dir=./ \
--examples_path=music.tfrecord
```
학습을 진행하면 root 폴더 안에 train이라는 폴더가 생기며, 그 안에 ckpt, meta data가 생성된다.  
또한 별도로 하이퍼 파라미터도 설정이 가능하다. 아래의 코드를 위의 코드와 연결하여 명령하면 된다.
```
--hparams=batch_size=32, learning_rate=0.0005
```

## Generate
위에서 train한 모델을 이용하여 midi 파일을 생성해준다.  

sample은 모델의 latent spaceing에서 임의의 지점을 디코딩하여 생성한다.  
interpolate는 더 창의적으로 만들기 위해서, 두 멜로디 사이사이에 삽입하는 동작을 평균하여 생성한다. 이때, 2파일의 구성요소와 bar의 길이는 같아야 한다.  

sample 방식을 사용할 경우 아래와 같이 실행한다.
```
python generate.py \
--config=groovae_4bar \
--mode=sample \ 
--num_outputs=1 \
--output_dir=output
``` 

interpolate 방식을 사용할 경우 아래와 같이 실행한다.  
```
python generate.py \
--config=groovae_4bar \
--mode=interpolate \
--num_outputs=1 \
--input_midi_1=$MIDI1_FILE_PATH \
--input_midi_2=$MIDI2_FILE_PATH \
--output_dir=output
```
`num_outputs`로 추출할 샘플의 갯수를 정한다.  
`output_dir`은 추출한 샘플이 저장될 장소를 의미한다.  

## Conclusion
### 진행한 내용
1. groovae 데이터를 이용하여 music VAE 모델을 학습 후, 생성해보았다.
2. 사전 학습된 `groovae_4bar` 모델을 사용하여 생성해보았다.

사전학습된 모델이 직접 데이터를 통한 학습한 것보다 더 다양한 노트를 사용하여 좋은 음원을 생성할 수 있었다.   
<p>
    <img width="251" alt="image" src="https://github.com/xdfc1745/music-VAE/assets/39234312/65610848-723e-4741-9a3e-9865d5595049">
    직접학습한 모델이 sample방식으로 생성한 midi 파일
</p>
<p>
  <img width="261" alt="image" src="https://github.com/xdfc1745/music-VAE/assets/39234312/846dd722-2900-4a67-918b-0f9083f91e8d">  
  사전학습된 모델이 sample방식으로 생성한 midi 파일
</p>
그에 반해, `interpolate`방식으로 생성한 midi파일의 경우 sample의 방식보다 더 다양한 노트를 사용한 파일을 생성해냈다.
<p>
  <img width="260" alt="image" src="https://github.com/xdfc1745/music-VAE/assets/39234312/e1028da6-79a7-4b4e-b6bd-8c23f38faa15">  
  생성한 모델 2개를 이용한 interpolate 방식으로 생성한 midi 파일
</p>

직접 생성한 모델의 경우 많은 epoch을 학습하지 못했고, 더 적은 데이터로 학습을 진행하였기 때문에 사전학습된 모델의 결과가 더 좋게 나온것으로 보인다.  
또한 sample기법에 비해 interpolate 기법이 더 좋은 결과를 보이는 것 같다.
