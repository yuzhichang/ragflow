import {
  KEYLESS_WEB_SEARCH_PROVIDERS,
  WebSearchProvider,
} from '@/constants/chat';
import type { PromptConfig } from '@/interfaces/database/chat';

export function getWebSearchProvider(promptConfig?: PromptConfig) {
  const provider = promptConfig?.web_search_provider;

  if (
    provider === WebSearchProvider.Brave ||
    provider === WebSearchProvider.Exa ||
    provider === WebSearchProvider.Firecrawl ||
    provider === WebSearchProvider.Linkup ||
    provider === WebSearchProvider.Parallel ||
    provider === WebSearchProvider.Querit ||
    provider === WebSearchProvider.Serply ||
    provider === WebSearchProvider.Tavily ||
    provider === WebSearchProvider.YouCom
  ) {
    return provider;
  }

  if (
    provider === undefined &&
    typeof promptConfig?.tavily_api_key === 'string' &&
    promptConfig.tavily_api_key.trim()
  ) {
    return WebSearchProvider.Tavily;
  }

  return undefined;
}

// The prompt_config field each provider reads its key from. Kept as the single
// source of truth: the form uses it for the required marker, the schema uses it
// for validation, and the reader below uses it to fetch the value — three
// callers that used to each spell the mapping out again.
const webSearchApiKeyFields: Record<WebSearchProvider, string> = {
  [WebSearchProvider.Brave]: 'brave_api_key',
  [WebSearchProvider.Exa]: 'exa_api_key',
  [WebSearchProvider.Firecrawl]: 'firecrawl_api_key',
  [WebSearchProvider.Linkup]: 'linkup_api_key',
  [WebSearchProvider.Parallel]: 'parallel_api_key',
  [WebSearchProvider.Querit]: 'querit_api_key',
  [WebSearchProvider.Serply]: 'serply_api_key',
  [WebSearchProvider.Tavily]: 'tavily_api_key',
  [WebSearchProvider.YouCom]: 'youcom_api_key',
};

export function getWebSearchApiKeyField(provider?: WebSearchProvider) {
  return provider ? webSearchApiKeyFields[provider] : undefined;
}

// Whether the selected provider must have a key before it can be used. A
// keyless provider answers on its own endpoint/tier and stays usable blank.
export function isWebSearchApiKeyRequired(provider?: WebSearchProvider) {
  return (
    provider !== undefined && !KEYLESS_WEB_SEARCH_PROVIDERS.includes(provider)
  );
}

export function getWebSearchApiKey(promptConfig?: PromptConfig) {
  const keyField = getWebSearchApiKeyField(getWebSearchProvider(promptConfig));
  if (!keyField) {
    return undefined;
  }

  const apiKey = (promptConfig as unknown as Record<string, unknown>)?.[
    keyField
  ];
  return typeof apiKey === 'string' ? apiKey.trim() : undefined;
}

/**
 * Whether web search is usable as configured. Most providers need a key; a
 * keyless provider is usable as soon as it is selected.
 */
export function hasWebSearchProvider(promptConfig?: PromptConfig) {
  const provider = getWebSearchProvider(promptConfig);

  if (provider === undefined) {
    return false;
  }
  if (KEYLESS_WEB_SEARCH_PROVIDERS.includes(provider)) {
    return true;
  }

  return Boolean(getWebSearchApiKey(promptConfig));
}
