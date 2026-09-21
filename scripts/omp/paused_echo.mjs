import nativeEcho from './native_echo.mjs';
import { createPausedEcho } from './paused_echo_control.mjs';
export default function pausedEcho(api) {
  const gate = createPausedEcho(process.env.DRIFT_PAUSED_ECHO_CONFIG, process.env.DRIFT_PAUSED_ECHO_SHA256);
  api.on('session_start', (_event, context) => gate.check(context));
  const wrapped = new Proxy(api, {
    get(target, key) {
      if (key === 'registerTool') return tool => {
        if (tool.name !== 'drift_native_echo') throw new Error('PAUSED_ECHO_CONTROL');
        target.registerTool({ ...tool, async execute(...args) { await gate.wait(args[4], args[2]); return tool.execute(...args); } });
      };
      const value = Reflect.get(target, key);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  nativeEcho(wrapped);
}
