import https from 'https';
import { ClientRequest, IncomingMessage } from 'http';
import { Socket } from 'net';
import { EventEmitter } from 'events';
import { Bot } from 'grammy';
import { afterEach, expect, it, vi } from 'vitest';
import { downloadTelegramFile } from './channels/telegram.js';

const sockets: Socket[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const socket of sockets.splice(0)) socket.destroy();
});
function response(status: number | undefined) {
  return vi.spyOn(https, 'get').mockImplementation((...args: unknown[]) => {
    const callback = args.find(
      (value): value is (message: IncomingMessage) => void =>
        typeof value === 'function',
    );
    if (!callback)
      throw new TypeError('Fixture expected the native response callback');
    const socket = new Socket();
    sockets.push(socket);
    const message = new IncomingMessage(socket);
    message.statusCode = status;
    queueMicrotask(() => {
      callback(message);
      message.push(Buffer.from('fixed provider body'));
      message.push(null);
    });
    return new EventEmitter() as ClientRequest;
  });
}
function botWithFile(filePath?: string) {
  const bot = new Bot('123:synthetic');
  vi.spyOn(bot.api, 'getFile').mockResolvedValue({
    file_id: 'fixed',
    file_unique_id: 'fixed',
    file_path: filePath,
  });
  return bot;
}
it.each([undefined, '', '   '])(
  'refuses missing or empty native file_path %j before any request',
  async (value) => {
    const get = response(200);
    await expect(
      downloadTelegramFile(botWithFile(value), 'fixed'),
    ).rejects.toThrow('no usable download path');
    expect(get).not.toHaveBeenCalled();
  },
);
it.each([undefined, 204, 206, 401, 404, 500])(
  'does not accept HTTP %s error/partial bodies as file bytes',
  async (status) => {
    response(status);
    await expect(
      downloadTelegramFile(botWithFile('documents/fixed.bin'), 'fixed'),
    ).rejects.toThrow('no file bytes accepted');
  },
);
it('accepts an ordinary complete HTTP200 file response through the actual helper', async () => {
  response(200);
  await expect(
    downloadTelegramFile(botWithFile('documents/fixed.bin'), 'fixed'),
  ).resolves.toEqual(Buffer.from('fixed provider body'));
});
