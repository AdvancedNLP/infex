import base64
import io
import os
from pprint import pprint
from typing import Optional

import httpx
import pdf2image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI


class Address(BaseModel):
    street: Optional[str] = Field(default=None, description='Address street')
    city: Optional[str] = Field(default=None, description='Address city')
    state: Optional[str] = Field(default=None, description='Address state')
    zip: Optional[str] = Field(default=None, description='Address zip code')


class CareProvider(BaseModel):
    name: Optional[str] = Field(default=None, description='The name of the care provider')
    phone: Optional[str] = Field(default=None, description='The phone number of the care provider')
    address: Address = Field(default=None, description='The address of the care provider')


class Service(BaseModel):
    date: Optional[str] = Field(default=None, description='The date the service was provided')
    quantity: int = Field(default=None, description='The quantity of the service')
    rate: Optional[str] = Field(default=None, description='The rate of the service')
    description: Optional[str] = Field(default=None, description='The description of the service')
    price: Optional[float] = Field(default=None, description='The price of the service')
    amount: Optional[float] = Field(default=None, description='The amount of the service')


class Invoice(BaseModel):
    care_provider: Optional[CareProvider] = Field(default=None, description='The care provider who provided the service')
    date: Optional[str] = Field(default=None, description='The date of the invoice')
    location: Optional[str] = Field(default=None, description='The location of the invoice')
    services: list[Service] = Field(default=None, description='The services provided by the care provider')
    total_amount: Optional[str] = Field(default=None, description='The total amount of the invoice')


def run_vision_extraction():
    image_data = _load_image('./0001.pdf')

    model = AzureChatOpenAI(
        azure_deployment='gpt-4o',
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
        openai_api_version="2023-12-01-preview",
        model_version="turbo-2024-04-09",
        temperature=0.0,
        http_client=httpx.Client(verify=False)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert extraction algorithm. Only extract relevant information from the text. "
                       "If you do not know the value of an attribute asked to extract, return null for the attribute's value."),
            ("human", [{"image_url": "data:image/jpeg;base64,{image_url}"}]),
        ]
    )

    structured_lmm = prompt | model.with_structured_output(schema=Invoice)

    invoice = structured_lmm.invoke(dict(image_url=image_data))
    pprint(invoice.dict())


def _load_image(content_file_path) -> str:
    images = pdf2image.convert_from_path(content_file_path)

    buffered = io.BytesIO()
    images[0].save(buffered, format="PNG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_data


if __name__ == '__main__':
    run_vision_extraction()
