# WebSockets Endpoint, API key, and connections

## How to get the API key and endpoint, connect to the WebSockets API, authenticate, create and resume sessions

## Using WebSockets

We currently support WebSocket connections as they are more efficient, faster, and less resource intensive. They are, however, a little harder to manage because of their async nature. We have however made our WebSocket connections easy to work with, as each response contains the request ID. So it's possible to easily match request → response.

> For your convenience we provide an asynchronous SDK to access the Runware api using javascript. See `@runware.ai/sdk-js` on npm.
>
> The project README.md is kept up to date and should support integrating the API using the SDK.

**WebSocket endpoint:** wss://ws-api.runware.ai/v1/

## New connections & authentication

WebSocket connections are point-to-point. So there's no need for each request to contain an authentication header. Instead, the first request must always be an authentication request that includes the API key. This way we can identify which subsequent requests are arriving from the same user.

```json
{
   "newConnection":{
      "apiKey":"<APIKEY>"
   }
}
```

## How to get the API key

To create an API key, simply sign up to Runware and visit the ‘API Keys’ page, then click to create a key. You can give it a name, a description, and create multiple keys if you need to for multiple environments (dev, prod, staging), and for different projects.

## WebSockets Session IDs

After you’ve made the authentication request the API will return a `connectionSessionUUID` in the following format:

```json
{
   "newConnectionSessionUUID":{
      "connectionSessionUUID":"f40c2aeb-f8a7-4af7-a1ab-7594c9bf778f"
   }
}
```

## Message buffer & resuming connections

If any service, server, or network is unresponsive, for instance due to a restart, all the images or tasks that could not be delivered are kept in a buffer memory for 2 minutes. It's possible to reconnect and have these messages delivered by providing in the initial authentication connection request the connectionSessionUUID, like in this example:

```json
{
   "newConnection":{
      "apiKey":"<APIKEY>",
      "connectionSessionUUID":"f40c2aeb-f8a7-4af7-a1ab-7594c9bf778f"
   }
}
```

After the connection is made it's possible to send different tasks to the API - e.g. text2img, img2img, inpainting, upscale, image2text, imageUpload, etc.
