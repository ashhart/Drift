let turn = 0;
export default function lifetimeFixture(api) {
  api.on("before_agent_start", event => {
    turn++;
    const system = Array.isArray(event.systemPrompt) ? event.systemPrompt : [event.systemPrompt ?? ""];
    return { systemPrompt: [...system, "Fixture own control " + turn] };
  });
  api.on("session_start", async () => api.setActiveTools([]));
}
