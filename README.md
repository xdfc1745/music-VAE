## music-VAE
Music genration을 위한 모델로, 문장과 달리 훨씬 긴 sequence를 생성하는 모델.  
sequence가 길어지면서 “posterior collapse”문제가 발생할 수 있는데 이를 해결하기 위해서 Hierarchical recurrent decoder를 제안.  
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
magenta의 convert_directory를 이용해서 벡터 파일인 TFRecord 파일로 변경해주었다. 이는 구글의 Protocla Buffer 포맷으로 데이터를 직렬화하여 저장한 파일 형태이다.  

https://magenta.tensorflow.org/datasets/groove   
데이터는 위의 링크에서 goovae-v1.0.0-midionly 파일을 다운받은 후, 압축을 풀어주었다.  

```
python preprocessing.py \
--input_dir=groove \
--output_file=music.tfrecord \
--recursive
```

## Train model
위에서 TFRecord 파일로 바꾼 데이터를 이용하여 모델을 학습시킨다.  

```
python train.py \
--config=groovae_4bar \ # cat-drum_2bar_small, cat-drum_2bar_big으로 바꿔서 사용가능
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
```
python generate.py \
--config=groovae_4bar \
--checkpoint_file=train \
--mode=sample \ 
--num_outputs=1 \
--output_dir=output
```
학습시 사용한 config와 같은 config로 지정해주어야한다.
