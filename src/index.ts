export const VERSION = '0.1.0';
export { InspectorBus, InspectorError, loopbackPair } from './bus';
export type { Transport, WireMessage } from './bus';
export { TensorStore } from './store';
export { mountPanel } from './panel/panel';
export type { PanelOptions, InspectorPanel } from './panel/panel';
export type * from './events';
