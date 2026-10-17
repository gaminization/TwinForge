import asyncio, json, sys, os, numpy as np
POSE=sys.argv[1]; os.makedirs(f'photos_{POSE}',exist_ok=True)
from playwright.async_api import async_playwright
views=[]
for ring,el in enumerate((18,38,60)):
    n=24
    for k in range(n): views.append((k*360/n+ring*7, el))
TARGET=(0.0,0.02,0.20); R=0.82
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path="/opt/pw-browsers/chromium",args=["--use-gl=swiftshader","--enable-unsafe-swiftshader","--no-sandbox"])
        pg=await b.new_page(viewport={"width":768,"height":1024})
        await pg.goto(f"http://127.0.0.1:8777/render2.html?pose={POSE}")
        await pg.wait_for_function("window.ready===true",timeout=60000)
        for i,(az,el) in enumerate(views):
            await pg.evaluate(f"view({az},{el},{R},{TARGET[0]},{TARGET[1]},{TARGET[2]})")
            await pg.screenshot(path=f"photos_{POSE}/img_{i:03d}.jpg",type="jpeg",quality=95)
        json.dump(dict(views=views,target=TARGET,radius=R),open(f"views_{POSE}.json","w"))
        print("rendered",len(views),"photos"); await b.close()
asyncio.run(main())
