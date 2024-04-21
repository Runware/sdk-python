# Prompt enhancer (Magic prompt)

## Can be used to add keywords to prompts that are meant to increase the quality or variety of results

Prompt enhancing can be used to attempt to generate different or better results for a particular topic. It works by adding keywords to a given prompt. Enhancing a prompt does not always retain the intended subject of the prompt and does not necessarily guarantee improved results over the original prompt.

Prompt enhancing requests must have the following format:

```json
{
    "newPromptEnhance": {
        "prompt": "a close up of a",
        "taskUUID": "38857cb7-92bc-4e3b-97ab-8d871d47a248",
        "promptMaxLength": 308,
        "promptVersions": 3,
        "promptLanguageId": 1
    }
}
```

| Parameter        | Type          | Use                                                                                                                                                  |
|------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| prompt           | string        | The prompt that you intend to enhance.                                                                                                               |
| taskUUID         | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task.                                                |
| promptMaxLength  | integer       | Character count. Represents the maximum length of the prompt that you intend to receive. Can take values between 1 and 380.                          |
| promptVersions   | integer       | The number of prompt versions that will be received. Can take values between 1 and 5.                                                                |
| promptLanguageId | integer       | The language prompt text. Can take values between 1 and 298. Default is 1 - English. Options are provided below.                                    |

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

Results will be generated in the following format:

```json
{
    "newPromptEnhancer": {
        "texts": [
            {
                "taskUUID": "75964cec-ff9f-485e-98e4-d1c4e5d3a814",
                "text": "a close up of a, very large epic camera lens, craig mullins, gopro lensed cityscape, dawn blue sky, very coherent asymmetrical artwork, cinematic, hyper realism, high detail, octane render, unreal engine, 8 k, depth of field"
            },
            {
                "taskUUID": "75964cec-ff9f-485e-98e4-d1c4e5d3a814",
                "text": "a close up of a, intricate, elegant, highly detailed,  digital painting, artstation, concept art, "
            },
            {
                "taskUUID": "75964cec-ff9f-485e-98e4-d1c4e5d3a814",
                "text": "a close up of a,"
            }
        ]
    }
}
```

An array of objects will be returned, where the count of objects depends on the amount of versions requested. Each object represents a text suggestion.

| Parameter | Type          | Use                                                                                     |
|-----------|---------------|-----------------------------------------------------------------------------------------|
| taskUUID  | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task. |
| text      | string        | The enhanced text/prompt response.                                                      |
