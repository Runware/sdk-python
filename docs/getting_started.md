# WebSockets Endpoint, API Key, and Connections

This guide explains how to authenticate, connect to, and interact with the Runware WebSocket API.

## Authentication

To interact with the Runware API, you need to authenticate your requests using an API key. This key is unique to your account and identifies you when making requests.

- You can create multiple keys for different projects or environments (development, production, staging).
- Keys can have descriptions and can be revoked at any time.
- With the new teams feature, you can share keys with your team members.

To create an API key:

1. Sign up on Runware
2. Visit the "API Keys" page
3. Click "Create Key"
4. Fill in the details for your new key

## WebSockets

We currently support WebSocket connections as they are more efficient, faster, and less resource-intensive. Our WebSocket connections are designed to be easy to work with, as each response contains the request ID, allowing for easy matching of requests to responses.

- The API uses a bidirectional protocol that encodes all messages as JSON objects.
- You can connect using one of our provided SDKs (Python, JavaScript, Go) or manually.
- If connecting manually, the endpoint URL is `wss://ws-api.runware.ai/v1`.

## New Connections

WebSocket connections are point-to-point, so there's no need for each request to contain an authentication header. Instead, the first request must always be an authentication request that includes the API key.

### Authentication Request

```json
[
  {
    "taskType": "authentication",
    "apiKey": "<YOUR_API_KEY>"
  }
]
```

### Authentication Response

On successful authentication, you'll receive a response with a `connectionSessionUUID`:

```json
{
  "data": [
    {
      "taskType": "authentication",
      "connectionSessionUUID": "f40c2aeb-f8a7-4af7-a1ab-7594c9bf778f"
    }
  ]
}
```

In case of an error, you'll receive an object with an error message:

```json
{
  "error": true,
  "errorMessageContentId": 1212,
  "errorId": 19,
  "errorMessage": "Invalid api key"
}
```

## Keeping Connection Alive

The WebSocket connection is kept open for 120 seconds from the last message exchanged. If you don't send any messages for 120 seconds, the connection will be closed automatically.

To keep the connection active, you can send a `ping` message:

```json
[
  {
    "taskType": "ping",
    "ping": true
  }
]
```

The server will respond with a `pong`:

```json
{
  "data": [
    {
      "taskType": "ping",
      "pong": true
    }
  ]
}
```

## Resuming Connections

If any service, server, or network becomes unresponsive, all undelivered images or tasks are kept in a buffer memory for 120 seconds. You can reconnect and receive these messages by including the `connectionSessionUUID` in the authentication request:

```json
[
  {
    "taskType": "authentication",
    "apiKey": "<YOUR_API_KEY>",
    "connectionSessionUUID": "f40c2aeb-f8a7-4af7-a1ab-7594c9bf778f"
  }
]
```

This means you don't need to resend the initial request; it will be delivered when reconnecting. SDK libraries handle reconnections automatically.

After establishing a connection, you can send various tasks to the API, such as text-to-image, image-to-image, inpainting, upscaling, image-to-text, image upload, etc.