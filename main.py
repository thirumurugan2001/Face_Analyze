from fastapi import FastAPI
import uvicorn
from model import UserFace
from searchImage import search_and_return_base64_results
app = FastAPI()


@app.post("/api/searchImage")
async def searchImage(Query: UserFace):
    try :
        print()
        response  = search_and_return_base64_results(Query.imageBase64, top_k=5)
        if response != None:
            return {
                "data":[{
                    "Name": response
                }],
                    "message":"Successfully completed the RAG for your Query",
                    "status":True,
                    "statusCode":200,
                },200
        else : 
            return {
                "data":[{
                    "Name": "No results found"
                }],
                "message":"Failed to RAG your Query",
                "status":False,
                "statusCode":400,
            },400
    except Exception as e:
        print("Error in the createResume",str(e))
        return {
            "data":[{
                    "response": str(e)
                }],
            "message":"Failed to RAG your Query",
            "stratusCode": 400,
            "status":False
        },400

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
