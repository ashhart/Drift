import { portEnv, secretEnv, serviceHost } from "./identity";
import { parseIntegerArgument } from "./util";

interface ConnectionSettings {
	host: string;
	port: number;
	secret: string;
}

export function connectionSettings(env: NodeJS.ProcessEnv = process.env): ConnectionSettings | string {
	const portText = env[portEnv]?.trim() ?? "";
	const secret = env[secretEnv] ?? "";
	const missing = [portText.length === 0 ? portEnv : undefined, secret.length === 0 ? secretEnv : undefined].filter(
		(name): name is string => name !== undefined,
	);
	if (missing.length > 0) {
		return "Drift needs " + missing.join(" and ") + " in the environment before it can reach the service.";
	}
	const port = parseIntegerArgument(portText);
	if (port === undefined || port < 1 || port > 65535) {
		return "Drift: " + portEnv + " must be a TCP port number (1-65535).";
	}
	return { host: serviceHost, port, secret };
}
