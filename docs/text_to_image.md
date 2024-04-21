# Text to Image

## Make image generation requests based on a given text prompt. Format, parameters, definitions and examples for text to image requests.

Text-to-image requests are the most well-known type of image generation request, based on a given prompt.

## Request Format for Text to Image

Requests must be sent in the following format:

```json
{
   "newTask":{
      "taskUUID":"string",
      "promptText":"string",
      "numberResults":int,
      "modelId": "string",
      "sizeId":int,
      "taskType":int,
      "promptLanguageId":null,
      "offset":int,
      "lora":[
         {
            "modelId":"string",
            "weight":float
         },
         {
            "modelId":"string",
            "weight":float
         }
      ],
      "controlNet":[
         {
            "preprocessor":"string",
            "weight":float,
            "startStep":int,
            "endStep":int,
            "guideImageUUID":"string",
            "controlMode": "string"
         },
          {
            "preprocessor":"string",
            "weight":float,
            "startStep":int,
            "endStep":int,
            "guideImageUUID":"string",
            "controlMode": "string"
         },
         {
            "preprocessor":"string",
            "weight":float,
            "startStep":int,
            "endStep":int,
            "guideImageUUID":"string",
            "controlMode": "string"
         }
      ]
   }
}
```

| Parameter | Type | Use |
| --- | --- | --- |
| `taskUUID` | UUIDV4 string | Used to identify the async responses to this task. It must be sent to match the response to the task. |
| `promptText` | string | Defines the prompt description of the image. Text can be in any language, as long as `promptLanguageId` is correctly defined to match. |
| `numberResults` | integer | The number of images to generate from the specified prompt |
| `sizeId` | integer | Predefined image sizes. Options are provided below. |
| `taskType` | integer | The ID of the task type. Can take values between 1 and 6. Options are provided below. For text to image use value: 1 |
| `promptLanguageId` | integer | The language prompt language ID. If `null` is sent, language will be automatically detected. Can take values between 1 and 298. Options are provided below. |
| `steps` | integer | The number of steps used to infer the image. Default is 20. Different step counts will provide different results. Can be experimented with but there is no direct relationship between higher step count and higher quality above default value. |
| `modelId` | string | The ID of the model used for the task. Options are provided below. |
| `gScale` | float | Guidance scale that takes float values between 0 and 10. Represent how closely the images will resemble the prompt or how much freedom the AI model has. Higher values are closer to the prompt. Default is 7.5. Low values may reduce the quality of the results. |
| `seed` | integer | Seed integer is used to randomize the image generation. Can take values from 1 to 922,337,203,470,729,216,000. If not set it will be randomly generated. If one seed is set but multiple images are generated the seed will be incremented by 1 (+1) for each image generated. |
| `useCache` | boolean | Optional. If true will return cached images generated with similar prompts and settings. If false will always return newly generated images. Default is true. |
| `saveToCache` | boolean | Optional. Default true. It cannot be disabled if no `uploadEndpoint` is defined. |
| `offset` | integer | Optional. If returning images from cache offset can be used to paginate across results. Default is 0. |
| `clipSkip` | integer | Optional. Used to skip some of the layers of the CLIP model. can get values from 0 to 2. |
| `styleId` | integer | Optional. Allows you to generate an image in a specific style. Here is a list of available styles |
| `uploadEndpoint` | string | Optional. Allows the image generated to be pushed using a `PUT` method to an endpoint of your choice (an S3 bucket, or any type of endpoint that would accept binary image data). The image will be pushed as binary data. |
| `checkNsfw` | bool | Optional. Default is false. Add an additional step to check if the image contains NSFW content. This adds an additional 0.1 seconds to image inference time. This check can rarely return false positives. |
| `usePromptWeighting` | bool | Optional. Allow setting different weights per words or expressions in prompts. If enabled, add 0.2 seconds to image inference time. Default is false. |
| `schedulerId` | integer | Optional. Allows the use of a specific scheduler. Defaults to scheduler associated with the selected model ID. List of scheduler IDs is provided below. |
| `lora` | array | Optional. If provided, should be an array of objects. Each object must have two attributes: `modelId` (string) and `weight` (float) with values from 0 to 1. E.g. `civitai:132942@146296`. The AIR stands for Artificial Intelligence Resource and can be obtained from the CivitAI website. To view the AIR it may be necessary to edit the CivitAI user settings. |
| `controlNet` | array | Optional. If provided, should be an array of objects. Each object must have five attributes: `preprocessor` (string) that could be one of the below options. `weight` (float) can have values between 0 and 1 and represent the weight of the ControlNet preprocessor in the image. `startStep` (integer) represents the moment in which the ControlNet preprocessor starts to control the inference. It can take values from 0 to the maximum number of `steps` in the image create request. This can also be replaced with `startStepPercentage` (float) which represents the same value but in percentages. It takes values from 0 to 1. `endStep` (integer) similar with `startStep` but represents the end of the preprocessor control of the image inference. The equivalent of the percentage option is `startStepPercentage` (float). `guideImageUUID` (UUIDV4 string) is the guideImageUUID. You can obtain this by uploading it using the image upload functionality. Instead of uploading the images first, it's possible to use a `guideImageUrl` and provide a link to a control image. Using this method however can be slower than uploading the image and retrieving the imageUUID. You can also create a guide image using the preprocessors. To do so follow this guide. `controlMode` has 3 options: `balanced`, `prompt`, `controlnet` |

## Image Sizes

Any image size can be used for any model. However, SDXL returns better results when using larger image sizes. Image sizes recommended for SDXL are labeled below.

```plaintext
1 // 512x square
'width' => 512,
'height' => 512

2 // 2:3 portrait
'width' => 512,
'height' => 768

3 // 1:2 portrait
'width' => 512,
'height' => 1024

4 // 2:3 landscape
'width' => 768,
'height' => 512

5 // 2:1 landscape
'width' => 1024,
'height' => 512

6 // 4:3 landscape
'width' => 704,
'height' => 512

7 // 16:9 landscape
'width' => 896,
'height' => 512

8 // 9:16 portrait
'width' => 512,
'height' => 896

9 // 3:4 portrait
'width' => 512,
'height' => 704

11 // 1024x square - SDXL
'width' => 1024,
'height' => 1024

16 // 16:9 landscape - SDXL
'width' => 1344,
'height' => 768

17 // 9:16 portrait - SDXL
'width' => 768,
'height' => 1344

20 // 2:3 portrait - SDXL
'width' => 640,
'height' => 960

21 // 3:2 landscape - SDXL
'width' => 960,
'height' => 640
```

## Task types

```plaintext
1  Text to Image
2  Image to Image
3  Inpainting
4  Image to Text
5  Prompt Enhancer
6  Image Upscale
7  Image Upload
8  Remove Background
9  ControlNet Text to Image
10 ControlNet Image to Image
11 ControlNet prepreocess image
```

## Model IDs

```plaintext
4 => SDXL v1.0
13 => rev_animated v1.2.2
18 => absolutereality v1.8.1
19 => cyberrealistic v3.2
20 => dreamshaper v7.0
22 => ghostmix_v2.0 bakedvae
25 => samaritan_3d_cartoon v3.0
```

## Language IDs

```plaintext
1 English (GB)
2 Afar
3 Abkhazian
4 Afrikaans
5 Amharic
6 Arabic
7 Assamese
8 Aymara
9 Azerbaijani
10 Bashkir
11 Belarusian
12 Bulgarian
13 Bihari
14 Bislama
15 Bengali/Bangla
16 Tibetan
17 Breton
18 Catalan
19 Corsican
20 Czech
21 Welsh
22 Danish
23 German
24 Bhutani
25 Greek
26 Esperanto
27 Spanish
28 Estonian
29 Basque
30 Persian
31 Finnish
32 Fiji
33 Faeroese
34 French
35 Frisian
36 Irish
37 Scots/Gaelic
38 Galician
39 Guarani
40 Gujarati
41 Hausa
42 Hindi
43 Croatian
44 Hungarian
45 Armenian
46 Interlingua
47 Interlingue
48 Inupiak
49 Indonesian
50 Icelandic
51 Italian
52 Hebrew
53 Japanese
54 Yiddish
55 Javanese
56 Georgian
57 Kazakh
58 Greenlandic
59 Cambodian
60 Kannada
61 Korean
62 Kashmiri
63 Kurdish
64 Kirghiz
65 Latin
66 Lingala
67 Laothian
68 Lithuanian
69 Latvian/Lettish
70 Malagasy
71 Maori
72 Macedonian
73 Malayalam
74 Mongolian
75 Moldavian
76 Marathi
77 Malay
78 Maltese
79 Burmese
80 Nauru
81 Nepali
82 Dutch
83 Norwegian
84 Occitan
85 (Afan)/Oromoor/Oriya
86 Punjabi
87 Polish
88 Pashto/Pushto
89 Portuguese
90 Quechua
91 Rhaeto-Romance
92 Kirundi
93 Romanian
94 Russian
95 Kinyarwanda
96 Sanskrit
97 Sindhi
98 Sangro
99 Serbo-Croatian
100 Singhalese
101 Slovak
102 Slovenian
103 Samoan
104 Shona
105 Somali
106 Albanian
107 Serbian
108 Siswati
109 Sesotho
110 Sundanese
111 Swedish
112 Swahili
113 Tamil
114 Telugu
115 Tajik
116 Thai
117 Tigrinya
118 Turkmen
119 Tagalog
120 Setswana
121 Tonga
122 Turkish
123 Tsonga
124 Tatar
125 Twi
126 Ukrainian
127 Urdu
128 Uzbek
129 Vietnamese
130 Volapuk
131 Wolof
132 Xhosa
133 Yoruba
134 Chinese
135 Zulu
136 Akan
137 Arabic (AE)
138 Arabic (BH)
139 Arabic (DZ)
140 Arabic (EG)
141 Arabic (IQ)
142 Arabic (JO)
143 Arabic (KW)
144 Arabic (LB)
145 Arabic (LY)
146 Arabic (MA)
147 Arabic (OM)
148 Arabic (QA)
149 Arabic (SA)
150 Arabic (SY)
151 Arabic (TN)
152 Arabic (YE)
153 Aragonese
154 Asturian
155 Avaric
156 Avestan
157 Azerbaijani (Cyrillic)
158 Bambara
159 Bengali
160 Bihari languages
161 Bosnian
162 Chamorro
163 Chechen
164 Chichewa
165 Chinese (HK)
166 Chinese (MO)
167 Chinese (SG)
168 Chinese (simplified)
169 Chinese (traditional)
170 Chinese (TW)
171 Church Slavic
172 Chuvash
173 Cornish
174 Cree
175 Dari
176 Divehi
177 Dzongkha
178 English (AU)
179 English (BZ)
180 English (CA)
181 English (GH)
182 English (HK)
183 English (IE)
184 English (IN)
185 English (JM)
186 English (KE)
187 English (MU)
188 English (NG)
189 English (NZ)
190 English (PH)
191 English (SG)
192 English (TT)
194 English (US)
195 English (ZA)
196 English (ZW)
197 Ewe
198 Faroese
199 Fijian
200 Filipino
201 Flemish
202 French (BE)
203 French (CA)
204 French (CH)
205 French (LU)
206 French (MC)
207 Fulah
208 Ganda
209 German (AT)
210 German (BE)
211 German (CH)
212 German (LI)
213 German (LU)
214 Haitian Creole
215 Herero
216 Hiri Motu
217 Ido
218 Igbo
219 Inuktitut
220 Inupiaq
221 Italian (CH)
222 Jamaican Patois
223 Kabyle
224 Kalaallisut
225 Kanuri
226 Khmer
227 Kikuyu; Gikuyu
228 Komi
229 Kongo
230 Kuanyama; Kwanyama
231 Lao
232 Latvian
233 Limburgish
234 Lojban
235 Luba-Katanga
236 Luxembourgish
237 Malay (BN)
238 Manx
239 Marshallese
240 Moldavian; Moldovan
241 Montenegrin
242 Montenegrin (Cyrillic)
243 Navajo; Navaho
244 Ndonga
245 North Ndebele
246 Northern Sami
247 Norwegian Bokmål
248 Norwegian Nynorsk
249 Ojibwa
250 Oriya
251 Oromo
252 Ossetian; Ossetic
253 Pali
254 Panjabi; Punjabi
255 Portuguese (BR)
256 Pushto; Pashto
257 Romani
258 Romansh
259 Rundi
260 Rusyn
261 Sango
262 Sardinian
263 Scottish Gaelic
264 Serbian (Cyrillic)
265 Sichuan Yi
266 Sicilian
267 Sinhalese
268 Sotho
269 South Ndebele
270 Spanish (AR)
271 Spanish (BO)
272 Spanish (CL)
273 Spanish (CO)
274 Spanish (CR)
275 Spanish (DO)
276 Spanish (EC)
277 Spanish (GT)
278 Spanish (HN)
279 Spanish (LA & C)
280 Spanish (MX)
281 Spanish (NI)
282 Spanish (PA)
283 Spanish (PE)
284 Spanish (PR)
285 Spanish (PY)
286 Spanish (SV)
287 Spanish (UY)
288 Spanish (VE)
289 Swati
290 Swedish (FI)
291 Tahitian
292 Tswana
293 Uighur
294 Uzbek (Cyrillic)
295 Venda
296 Walloon
297 Western Frisian
298 Zhuang; Chuang
```

## ControlNet PreProcessors

SDXL controlnet has (at this moment) less number of preprocessor types

```plaintext
SD 1.5

canny
depth_leres
depth_midas
depth_zoe
inpaint_global_harmonious
lineart_anime
lineart_coarse
lineart_realistic
lineart_standard
mlsd
normal_bae
openpose
openpose_face
openpose_faceonly
openpose_full
openpose_hand
scribble_hed
scribble_pidinet
seg_ofade20k
seg_ofcoco
seg_ufade20k
shuffle
softedge_hed
softedge_hedsafe
softedge_pidinet
softedge_pidisafe
tile_gaussian

SDXL

canny
depth_leres
depth_midas
depth_zoe
inpaint_global_harmonious
```

## Scheduler IDs

```plaintext
1 => "Default" //default model scheduler
2 => "DDIMScheduler"
3 => "DDIMInverseScheduler"
4 => "DDPMScheduler"
5 => "DEISMultistepScheduler"
6 => "DPMSolverSinglestepScheduler"
7 => "DPMSolverMultistepScheduler"
8 => "HeunDiscreteScheduler"
9 => "KDPM2DiscreteScheduler"
10 => "KDPM2AncestralDiscreteScheduler"
11 => "KarrasVeScheduler"
12 => "LMSDiscreteScheduler"
13 => "PNDMScheduler"
14 => "ScoreSdeVeScheduler"
15 => "IPNDMScheduler"
16 => "ScoreSdeVpScheduler"
17 => "EulerDiscreteScheduler"
18 => "EulerAncestralDiscreteScheduler"
19 => "VQDiffusionScheduler"
20 => "UniPCMultistepScheduler"
21 => "RePaintScheduler"
22 => "DPM++ 2M Karras"
23 => "DPM++ 2M SDE Karras"
24 => "DPM++ 2M SDE"
25 => "DPM++ SDE Karras"
26 => "DPM++ SDE"
```

## Style IDs

```plaintext
1 => "No style"
2 => "Anime"
3 => "Photographic"
4 => "Digital Art"
5 => "Comic Book"
6 => "Fantasy Art"
7 => "Analog Film"
8 => "Neon Punk"
9 => "Isometric"
10 => "Low Poly"
11 => "Origami"
12 => "Line Art"
```

## Results format

Results will be delivered in the format below. It's possible to receive one or multiple images per message. This is due to the fact that images are generated in parallel, and generation time varies across nodes or the network.

```json
{
    "newImages": {
        "images": [
            {
                "imageSrc": "https://im.runware.ai/image/ii/e1e7b7b8-046d-48f6-b09b-381d67bef00d.jpg",
                "imageUUID": "87e9f0eb-ab36-4c17-95ca-4ae4ac48e383",
                "bNSFWContent": false,
                "imageAltText": "country house on top of a hill, idyllic, french countryside, beautiful, nature, warm lighting, crisp detail, high definition",
                "taskUUID": "99e54383-551d-4029-9928-bc83177e26ea"
            },
            {
                "imageSrc": "https://im.runware.ai/image/ii/216877ea-f993-4333-a4c5-c97f534637fc.jpg",
                "imageUUID": "ba45e405-6042-4796-896a-105c1ab72cf2",
                "bNSFWContent": false,
                "imageAltText": "country house on top of a hill, idyllic, french countryside, beautiful, nature, warm lighting, crisp detail, high definition",
                "taskUUID": "99e54383-551d-4029-9928-bc83177e26ea"
            }
        ],
        "totalAvailableResults": 18
    }
}
```

Results will be received as an array of objects:

| Parameter             | Type          | Use                                                                                                                                                                |
|-----------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `imageSrc`              | string        | The URL of the image to be downloaded.                                                                                                                             |
| `imageUUID`             | UUIDv4 string | The unique identifier of the image.                                                                                                                                |
| `bNSFWContent`          | boolean       | Used to inform if the image has been flagged as potentially sensitive content. True indicates the image has been flagged (is a sensitive image). False indicates the image has not been flagged. The filter occasionally returns false positives and very rarely false negatives. |
| `imageAltText`          | string        | If sensitive words are used in the prompt these are filtered by default and are not added to the alt text. It may be missing if the images are newly generated. It is returned just from images from the cache |
| `taskUUID`              | UUIDv4 string | Used to match the async responses to their corresponding tasks.                                                                                                     |
| `totalAvailableResults` | integer       | Total number of existing images (in cache) available for the provided prompt.                                                                                      |
