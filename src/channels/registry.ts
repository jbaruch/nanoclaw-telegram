import {
  Channel,
  OnInboundMessage,
  OnChatMetadata,
  OnLocation,
  RegisteredGroup,
} from '../types.js';

export interface ChannelOpts {
  onMessage: OnInboundMessage;
  onChatMetadata: OnChatMetadata;
  // Optional — channels that don't deliver location/venue payloads omit
  // it. Telegram provides static pins, venues, and live-location ticks;
  // other channels (WhatsApp, Slack) may follow later.
  onLocation?: OnLocation;
  registeredGroups: () => Record<string, RegisteredGroup>;
}

export type ChannelFactory = (opts: ChannelOpts) => Channel | null;

const registry = new Map<string, ChannelFactory>();

export function registerChannel(name: string, factory: ChannelFactory): void {
  registry.set(name, factory);
}

export function getChannelFactory(name: string): ChannelFactory | undefined {
  return registry.get(name);
}

export function getRegisteredChannelNames(): string[] {
  return [...registry.keys()];
}
