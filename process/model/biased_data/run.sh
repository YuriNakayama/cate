python process/model/biased_data \
    --multirun \
    data.name=hillstorm,megafon,lenta,criteo \
    model.name=drlearner,xlearner,rlearner,slearner,tlearner \
    data.random_ratio=0.0,0.2,0.4,0.6,0.8,1.0

python process/model/biased_data \
    --multirun \
    data.name=hillstorm \
    model.name=drlearner,xlearner,rlearner,slearner,tlearner \
    data.random_ratio=0.0,0.2,0.4,0.6,0.8,1.0\
    random_state=0,1,2,3,4,5,6,7,8,9
