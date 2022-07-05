from pydantic import BaseModel, confloat

class RequestData(BaseModel):
    sepal_length: confloat(ge=0.0)
    sepal_width: confloat(ge=0.0)
    petal_length: confloat(ge=0.0)
    petal_width: confloat(ge=0.0)

class InferData(BaseModel):
    iris_setosa: confloat(ge=0.0, le=1.0)
    iris_versicolour: confloat(ge=0.0, le=1.0)
    iris_virginica: confloat(ge=0.0, le=1.0)