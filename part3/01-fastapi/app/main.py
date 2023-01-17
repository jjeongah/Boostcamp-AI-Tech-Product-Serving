from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.model import MyEfficientNet, get_model, get_config, predict_from_image_byte

app = FastAPI()

orders = [] #💚 실무에서는 보통 데이터 베이스를 이용해 주문하지만, 위 실습에서는 In memory인 리스트에 저장


@app.get("/")
def hello_world():
    return {"hello": "world"}


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4) 
    # UUID: 고유식별자 
    # Field : 모델 스키마 또는 복잡한 validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # dafault_factory : Product class가 처음 만들어질 때 호출되는 함수를 uuid4로 하겠다 / product class를 생성하면 uuid4를 만들어서 id에 저장
    name: str
    price: float


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product]  = Field(default_factory=list)
    # 최초의 빈 List를 만들어 저장한다
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


class OrderUpdate(BaseModel): #💛
    products: List[Product] = Field(default_factory=list)


class InferenceImageProduct(Product): #🧡
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders #💚


@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    # order_id를 기반으로 order을 가져온다
    order = get_order_by_id(order_id=order_id)
    if not order:
        # 만약 get_order_by_id에서 아무런 데이터가 없어 빈 리스트가 나온다면?
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    # generator (iter, next 키워드)
    # generator을 사용한 이유: 메모리를 절약해서 사용 가능
    # iter: 반복 가능한 객체에서 iterator을 반환 / next: iterator에서 값을 차례대로 꺼냄
    return next((order for order in orders if order.id == order_id), None) #💚


@app.post("/order", description="주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MyEfficientNet = Depends(get_model),
                     config: Dict[str, Any] = Depends(get_config)):
    # Depends: 의존성 주입. 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # Model을 Load, Config을 Load
    products = []
    for file in files:
        image_bytes = await file.read()
        inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
        product = InferenceImageProduct(result=inference_result) #🧡
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order) #💚
    return new_order


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]: #💛
    """
    Order를 업데이트 합니다

    Args:
        order_id (UUID): order id
        order_update (OrderUpdate): Order Update DTO

    Returns:
        Optional[Order]: 업데이트 된 Order 또는 None
    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)

    return updated_order


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)

    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill
