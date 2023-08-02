<div>
    <center>
        <img src="https://s1.ax1x.com/2023/08/03/pPiELSs.png" alt="pPiELSs.png" style="zoom:20%;" />				<h1>MoeGoeApi</h1>
    <center>
</div>

# What's this

**MoeGoeAPI** is based on VITS local reasoning project [MoeGoe ](https://github.com/CjangCjengh/MoeGoe)version.

It can help us ask for language generation on different languages or clients, as long as you have VITS model.

I'm just doing an Electron project and needed to ask for language generation, so I found the MoeGoe project, and joined the Flask framework so that it can be requested in the form of API so that I can ask the language in Nodejs to generate the language generation to generate the language generation.

## How to use
You only need to add your model file to the Module folder, and rename the model to **G_13000.pth**, and name the configuration file **config.json**.(Or you can change this heavy naming rule in moegoe.py).

And then use Anaconda or Venv to create a virtual environment, install all the packages in requirements.txt, and then enter **python Moegoe.py**.

**/generate_Audio** is where you send a request.（post）The returned audio file will be stored in the Voice folder under the project folder

**/audio/[filename]** is where you receive audio.[filename] is the name of the audio file requested to return, named after the timestamp（get）

You can use Postman to post this api：Enter the following format in body

{

 "text": "こんにちは,灰の魔女イレーヌです"

}

