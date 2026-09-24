declare module "*subagent_command.mjs" {
  export function installSubagentCommand(api: import("../src/types").OmpExtensionApi):
    (args: string, context: import("../src/types").CommandContext) => Promise<void>;
}
