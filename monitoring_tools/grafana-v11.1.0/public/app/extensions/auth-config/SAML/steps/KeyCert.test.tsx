import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { TestProvider } from 'test/helpers/TestProvider';

import { getRouteComponentProps } from 'app/core/navigation/__mocks__/routeProps';
import { addRootReducer } from 'app/store/configureStore';

import { mockToolkitActionCreator } from '../../../../../test/core/redux/mocks';
import {
  authConfigSAMLReducer,
  settingsUpdated,
  setSignRequests,
  setKeyCertValueType,
  setKeyConfigured,
  setCertConfigured,
} from '../state/reducers';

import { KeyCertUnconnected, Props } from './KeyCert';

beforeEach(() => {
  jest.clearAllMocks();
});

jest.mock('app/core/core', () => {
  return {
    contextSrv: {
      hasPermission: () => true,
    },
  };
});

jest.mock('@grafana/runtime/src/config', () => ({
  config: {
    ...jest.requireActual('@grafana/runtime/src/config').config,
    buildInfo: {
      edition: 'Enterprise',
      version: '10.0.0',
      commit: 'abcdefa',
      env: 'dev',
      latestVersion: '',
      hasUpdate: false,
      hideVersion: false,
    },
    licenseInfo: {
      enabledFeatures: { saml: true },
    },
    featureToggles: {
      accesscontrol: true,
    },
    bootData: { navTree: [], user: {} },
  },
}));

const defaultSettings = {
  privateKey: '',
  privateKeyPath: '',
  certificate: '',
  certificatePath: '',
  signatureAlgorithm: 'rsa-sha256',
  keyCertValueType: 'base64',
  signRequests: false,
};

const mockUpdate = mockToolkitActionCreator(settingsUpdated);

const setup = (propOverrides?: Partial<Props>) => {
  addRootReducer(authConfigSAMLReducer);
  const props: Props = {
    ...getRouteComponentProps(),
    samlSettings: defaultSettings,
    storedSettings: {},
    signRequests: false,
    keyCertValueType: 'base64',
    keyConfigured: false,
    certConfigured: false,
    setSignRequests: mockToolkitActionCreator(setSignRequests),
    setKeyCertValueType: mockToolkitActionCreator(setKeyCertValueType),
    setKeyConfigured: mockToolkitActionCreator(setKeyConfigured),
    setCertConfigured: mockToolkitActionCreator(setCertConfigured),
    settingsUpdated: mockUpdate,
    ...propOverrides,
  };

  render(
    <TestProvider>
      <KeyCertUnconnected {...props} />
    </TestProvider>
  );

  return {
    user: userEvent.setup(),
  };
};

describe('StepKeyCert', () => {
  it('should render with default values', () => {
    setup();

    expect(screen.getByRole('checkbox', { name: /Sign requests/i })).not.toBeChecked();
    expect(screen.queryByText(/Signature algorithm/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Next/i })).toBeInTheDocument();
  });

  it('should show signature algorithm if sign request enabled', async () => {
    const { user } = setup({ settingsUpdated: mockUpdate });
    await user.click(screen.getByRole('checkbox', { name: /Sign requests/i }));

    expect(screen.getByText(/^Signature algorithm$/i)).toBeInTheDocument();
  });

  it('should submit correct form data', async () => {
    const { user } = setup({ settingsUpdated: mockUpdate });

    await user.click(screen.getByRole('checkbox', { name: /Sign requests/i }));
    await user.click(screen.getByRole('radio', { name: /RSA-SHA512/i }));
    await user.type(
      screen.getByLabelText(/Private key/),
      'LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUV2Z0lCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktnd2dnU2tBZ0VBQW9JQkFRQzNDa3ZEMEtmdXh0V2gKNFFHV1cxa254UHpXYlNiZWpKOTNvQUZPV2JMNXFvM01FT0FJdkl4NFVndEcvelk3dW03aEtiZnh0OVg0OEVDTwp1MFFvQmVLek40Q2U2L0JhSVhrYnYwaHl1MEhNVy9mQ1JUbFhPWDIrcFIvTm9HTVZ4QW5FL3hRL0dmUUlMRlVJCmQ2dHkwQS9tM0x2c0xOQS9UcUQwM1UzazFZNzIwdmZJWU1ZZkhPR3dSOGFGSFB4OEhINkRYbGFVV21lQjJZZVEKMXZFaTVaNzhONXlxZUx4KzlnQXhMMWdnQWJVZ1g2RWFCcnBpWE9ndFF5NFhRVzJ2a0J0Z1RtMUc3eFhGYks3UQpSWGRpa1dVN0JwNk44YnpXeWM3SWUvWE5USm0xOVRRSlBXOC9sTGlrZ21mUFZSQ0ZNZisvQzJwQjhlNkE4S0F6CnVUL25rS3ZmQWdNQkFBRUNnZ0VCQUtIc1Fjb2dNKzlQaG4yR0IxbHp4aE1nZUd0WWRDSThWSFlHM2lEcHlTS0UKZHB3bjhZakpBNXNOT1Z5OU9NVHQwRmp3QndHV2wzdDVUaUwyRnYwdzk3T0RUZzFJdTMrQVRScjJ4S2g2YjJMUgptUVFMS1FhNG5UeWg1dGlKSWg2SEpmMVhpaG1vWS9FM0JxSk8reVMrV3RMb1FuWlR6b1d5a1VFTTVYT2lXSzZQCncwaVdQNFlWOWg1UWx6d3F2QWRpUExnb1NvT0FvNzZ2K2M2SGQxWXdUbnVPenE2U2RaeEZvSTRtT0JlczlhV20KOFgxMlAzdFRGb2szZEE5UXpRYkdaalRJS3hDbzNlMk5oaEJYM2xheTZGS0F2LzNyRDVzY2JCdnl2Wk1ZTk9LaQpiOUs0UzVyK3FTaTFSNGF6NmhXSWFxaXBwSHFnK1ZicmZXMnFZaHNRWkVFQ2dZRUE1dWF4KzA5YXFyOHhUUzg3CjFjRkNESThUdTRJWnpVTTgxY0dubVRib3FCMGlkb0lLWWxsMXJwdU9YUWlvREZGQktTMHhpd005U0hNRHZLQWMKTHRnMmFJcWdQODdiciswbWkzSmNKUENwTzhKcTlNTVZZSEU4OEFaVFNpbGdIUkZUTDZYeXFOVUp1Q01CV3BLcApyMUNXM25Nem9XV0wyNnlRWm56Q1lWa0VCNjBDZ1lFQXl1L0ZTekxiTk1TZWpubGZWTWpIYkJScVBzQWJudUFICjNrcEFZU2JwYUxUcXBLc094NWE5SUlCZVhaOHdEdlVTdFlNdFRJc0ZaNVBBbzMyeHlNbDg1VVBDbUpzaTExREoKbWdobUdIdW1ySUM2WW94QVlIZVZZd3g4MTlkbTdGMElQSm80elZOS0JXSXRnZE85TnlDSHhXbEk4VzR5TXZEQgpmYjdYdHJpM1l6c0NnWUVBcUFob3E3ZVRCc0Ura3FuMkJGZXd4a3oxRWpiYXFncnRZYTJkVngyamNIeExVU3pQCmU4VVlwQ25tRDRRMkwwaU9mL1c0b1M2SlVZa0p4aUpLZ0RpdVY2UFNmTDBBR09ZZmI4bVlqdlZvWXVuWVJpa1cKREdvNzYzemU3aEJ3TnBjZUQ0WFhzcy9lVTlkOXE3a2hyVWlia21VSnhNR09aeDRNczlqRTZHME1ScjBDZ1lCWApxL2huUVdSVVNyRlpTZlEyYzdzNzVTSDFWQmRhcnR5WFd5cXZEaGwrWTlLUEsxdHFQS1FlVzNicEtYR2d0SjNOCmp3dE42eTE5anVhRmV6TUlWSXZCbkdHQWZsMHgralB1ZUx4cXpWaTNZN0dpQzRkd1pabERQa3pkRHJEbllsWEIKb0xyemNZMGtBRU02dUZnaGJjdGFMTjhYZG96M1Z1Mnd2bmNBK0pJNXF3S0JnQjczQVVtVmVieG9xRzdGK0NlbQpnSDJ1RDYwanVzVzRtL0JvVVJGMmtibXBLMXBJNFRQSE1XZVJ4UnpSNmJXS0lrZUgwdGJxNHdIbnhjd01IZTFECkdqSmRZYS90ZmtmWGRiYWZpRnZvankxQWRuU3NKZm1SOGtwRlFzTnNtb25CZExhZDVuQndCOWlmWEVMWGZKK2UKVGZmWTIwT28rUjQ2V0lrYzlZMXE4QjhjCi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0K'
    );
    await user.type(
      screen.getByRole('textbox', { name: /^Certificate/i }),
      'LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURhekNDQWxPZ0F3SUJBZ0lVVzNzY09HUUE5a3FEb1Nja2hsTjh0NFhkZTRvd0RRWUpLb1pJaHZjTkFRRUwKQlFBd1JURUxNQWtHQTFVRUJoTUNRVlV4RXpBUkJnTlZCQWdNQ2xOdmJXVXRVM1JoZEdVeElUQWZCZ05WQkFvTQpHRWx1ZEdWeWJtVjBJRmRwWkdkcGRITWdVSFI1SUV4MFpEQWVGdzB5TXpBM01URXhNelEyTURCYUZ3MHlOREEzCk1UQXhNelEyTURCYU1FVXhDekFKQmdOVkJBWVRBa0ZWTVJNd0VRWURWUVFJREFwVGIyMWxMVk4wWVhSbE1TRXcKSHdZRFZRUUtEQmhKYm5SbGNtNWxkQ0JYYVdSbmFYUnpJRkIwZVNCTWRHUXdnZ0VpTUEwR0NTcUdTSWIzRFFFQgpBUVVBQTRJQkR3QXdnZ0VLQW9JQkFRQzNDa3ZEMEtmdXh0V2g0UUdXVzFrbnhQeldiU2Jlako5M29BRk9XYkw1CnFvM01FT0FJdkl4NFVndEcvelk3dW03aEtiZnh0OVg0OEVDT3UwUW9CZUt6TjRDZTYvQmFJWGtidjBoeXUwSE0KVy9mQ1JUbFhPWDIrcFIvTm9HTVZ4QW5FL3hRL0dmUUlMRlVJZDZ0eTBBL20zTHZzTE5BL1RxRDAzVTNrMVk3MgowdmZJWU1ZZkhPR3dSOGFGSFB4OEhINkRYbGFVV21lQjJZZVExdkVpNVo3OE41eXFlTHgrOWdBeEwxZ2dBYlVnClg2RWFCcnBpWE9ndFF5NFhRVzJ2a0J0Z1RtMUc3eFhGYks3UVJYZGlrV1U3QnA2Tjhield5YzdJZS9YTlRKbTEKOVRRSlBXOC9sTGlrZ21mUFZSQ0ZNZisvQzJwQjhlNkE4S0F6dVQvbmtLdmZBZ01CQUFHalV6QlJNQjBHQTFVZApEZ1FXQkJTd0ovV1VjeE5NOVZlbjhxL3d6eUN3bG15aFBUQWZCZ05WSFNNRUdEQVdnQlN3Si9XVWN4Tk05VmVuCjhxL3d6eUN3bG15aFBUQVBCZ05WSFJNQkFmOEVCVEFEQVFIL01BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQ2YKTVNOa01WZE1NT054ZDF1azZoU2lwRFpYSm4zMXN0UlBuZWRRQ0twQ2t0ZXN4RmpkanlTemdaWVRzZm5SZHR2TgpWVVVBa1RzOEZDa3ZJMS8wY21OKzlLNlRLZWR2WnNSMktjcGR3VUhDeHZEZEd3Um1Udnh2MDhvd3BNR2lGZm1tClZNYk5PZ2NNN0hrdy9McDliL1d5dmJaejNPTGNlbDJvb1VvaHRlSkhNcW96dlYyZkVkcExEcWdiNktzN1l4SDgKb2U0QWZyNXlNOTU5WDJ2b2g1Z3h4bHFIWjdoemJKWTlibml2bEJrN1preEdLREpSMVNKeG45WDd6RisybUljNgozQUlWYUtnUkt5SkE0QzRYN0VydnA5WG1WUzAvMDMrK09pOXFRZHVkV0VYc0YwNVN4N1NXTmp3amQ1cm1jOU9oCnMxdEFFNktFa2pObE9XdHVlZTZxCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K'
    );

    await user.click(screen.getByRole('button', { name: /^Next/i }));

    await waitFor(() =>
      expect(mockUpdate).toHaveBeenCalledWith({
        ...defaultSettings,
        signatureAlgorithm: 'rsa-sha512',
        /*
        -----BEGIN PRIVATE KEY-----
MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8AgEAAkEA2nLedzz7vCg2BdZT
+YUtcTzeTZ1gqvzZpClA8wVrBd2jMOOL/lyokTBvrdvBjEiRmN7dE8A8SzsHnkk3
ySJ8IwIDAQABAkEAq/mkv5AeA9K31ZjIfIJKkmuTdZkHmQ2nFLcrLUgUZ/xYeTkN
69B3SDl8HJfCi76VaJXQUAMJ7zLt0y8sMStpAQIhAPdp8yS7jO9GAy27R1SEYVvV
aBjGGvmw1eaPkJLfZ/8xAiEA7jXZIOlO2eG3XC1ZJIURC6VWa8Nv+UMCEZ9R23XC
4nUCIQDLN7BO2VofeFJhlvNVPG8+1Taw+6b5gLNyHYD9IjvqIQIhAIhBr7wzS+Gm
X0Th1jJIe9YJlaTR/lInHHnnvboHJ3xBAiEAuU1N+Hkr6H9a2hD6Ilv2geCbp5LB
yHlvMjOBiJi6CaY=
-----END PRIVATE KEY-----
        */
        privateKey:
          'LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUV2Z0lCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktnd2dnU2tBZ0VBQW9JQkFRQzNDa3ZEMEtmdXh0V2gKNFFHV1cxa254UHpXYlNiZWpKOTNvQUZPV2JMNXFvM01FT0FJdkl4NFVndEcvelk3dW03aEtiZnh0OVg0OEVDTwp1MFFvQmVLek40Q2U2L0JhSVhrYnYwaHl1MEhNVy9mQ1JUbFhPWDIrcFIvTm9HTVZ4QW5FL3hRL0dmUUlMRlVJCmQ2dHkwQS9tM0x2c0xOQS9UcUQwM1UzazFZNzIwdmZJWU1ZZkhPR3dSOGFGSFB4OEhINkRYbGFVV21lQjJZZVEKMXZFaTVaNzhONXlxZUx4KzlnQXhMMWdnQWJVZ1g2RWFCcnBpWE9ndFF5NFhRVzJ2a0J0Z1RtMUc3eFhGYks3UQpSWGRpa1dVN0JwNk44YnpXeWM3SWUvWE5USm0xOVRRSlBXOC9sTGlrZ21mUFZSQ0ZNZisvQzJwQjhlNkE4S0F6CnVUL25rS3ZmQWdNQkFBRUNnZ0VCQUtIc1Fjb2dNKzlQaG4yR0IxbHp4aE1nZUd0WWRDSThWSFlHM2lEcHlTS0UKZHB3bjhZakpBNXNOT1Z5OU9NVHQwRmp3QndHV2wzdDVUaUwyRnYwdzk3T0RUZzFJdTMrQVRScjJ4S2g2YjJMUgptUVFMS1FhNG5UeWg1dGlKSWg2SEpmMVhpaG1vWS9FM0JxSk8reVMrV3RMb1FuWlR6b1d5a1VFTTVYT2lXSzZQCncwaVdQNFlWOWg1UWx6d3F2QWRpUExnb1NvT0FvNzZ2K2M2SGQxWXdUbnVPenE2U2RaeEZvSTRtT0JlczlhV20KOFgxMlAzdFRGb2szZEE5UXpRYkdaalRJS3hDbzNlMk5oaEJYM2xheTZGS0F2LzNyRDVzY2JCdnl2Wk1ZTk9LaQpiOUs0UzVyK3FTaTFSNGF6NmhXSWFxaXBwSHFnK1ZicmZXMnFZaHNRWkVFQ2dZRUE1dWF4KzA5YXFyOHhUUzg3CjFjRkNESThUdTRJWnpVTTgxY0dubVRib3FCMGlkb0lLWWxsMXJwdU9YUWlvREZGQktTMHhpd005U0hNRHZLQWMKTHRnMmFJcWdQODdiciswbWkzSmNKUENwTzhKcTlNTVZZSEU4OEFaVFNpbGdIUkZUTDZYeXFOVUp1Q01CV3BLcApyMUNXM25Nem9XV0wyNnlRWm56Q1lWa0VCNjBDZ1lFQXl1L0ZTekxiTk1TZWpubGZWTWpIYkJScVBzQWJudUFICjNrcEFZU2JwYUxUcXBLc094NWE5SUlCZVhaOHdEdlVTdFlNdFRJc0ZaNVBBbzMyeHlNbDg1VVBDbUpzaTExREoKbWdobUdIdW1ySUM2WW94QVlIZVZZd3g4MTlkbTdGMElQSm80elZOS0JXSXRnZE85TnlDSHhXbEk4VzR5TXZEQgpmYjdYdHJpM1l6c0NnWUVBcUFob3E3ZVRCc0Ura3FuMkJGZXd4a3oxRWpiYXFncnRZYTJkVngyamNIeExVU3pQCmU4VVlwQ25tRDRRMkwwaU9mL1c0b1M2SlVZa0p4aUpLZ0RpdVY2UFNmTDBBR09ZZmI4bVlqdlZvWXVuWVJpa1cKREdvNzYzemU3aEJ3TnBjZUQ0WFhzcy9lVTlkOXE3a2hyVWlia21VSnhNR09aeDRNczlqRTZHME1ScjBDZ1lCWApxL2huUVdSVVNyRlpTZlEyYzdzNzVTSDFWQmRhcnR5WFd5cXZEaGwrWTlLUEsxdHFQS1FlVzNicEtYR2d0SjNOCmp3dE42eTE5anVhRmV6TUlWSXZCbkdHQWZsMHgralB1ZUx4cXpWaTNZN0dpQzRkd1pabERQa3pkRHJEbllsWEIKb0xyemNZMGtBRU02dUZnaGJjdGFMTjhYZG96M1Z1Mnd2bmNBK0pJNXF3S0JnQjczQVVtVmVieG9xRzdGK0NlbQpnSDJ1RDYwanVzVzRtL0JvVVJGMmtibXBLMXBJNFRQSE1XZVJ4UnpSNmJXS0lrZUgwdGJxNHdIbnhjd01IZTFECkdqSmRZYS90ZmtmWGRiYWZpRnZvankxQWRuU3NKZm1SOGtwRlFzTnNtb25CZExhZDVuQndCOWlmWEVMWGZKK2UKVGZmWTIwT28rUjQ2V0lrYzlZMXE4QjhjCi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0K',
        /*
        -----BEGIN CERTIFICATE-----
MIIBWjCCAQOgAwIBAgIQVNRhWtyTmsNGwpKZoFnh+jAKBggqhkjOPQQDAjASMRAw
DgYDVQQKEwdBY21lIENvMB4XDTIxMTAyMzIyMjEzMFoXDTIyMTAyMzIyMjEzMFow
EjEQMA4GA1UEChMHQWNtZSBDbzBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABO/X
lA4ow3B7uR/hC8BB6JhQjt6OZFG25epGed7l8RHWTWj70Z4YpjnQH5Kc1c1ceC7H
YsAaScB+rxqGzHgR3XOjbzBtMA4GA1UdDwEB/wQEAwIFoDATBgNVHSUEDDAKBggr
BgEFBQcDATAMBgNVHRMBAf8EAjAAMBYGA1UdEQQPMA2CD2FjbWUtY2EuZX
        */
        certificate:
          'LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURhekNDQWxPZ0F3SUJBZ0lVVzNzY09HUUE5a3FEb1Nja2hsTjh0NFhkZTRvd0RRWUpLb1pJaHZjTkFRRUwKQlFBd1JURUxNQWtHQTFVRUJoTUNRVlV4RXpBUkJnTlZCQWdNQ2xOdmJXVXRVM1JoZEdVeElUQWZCZ05WQkFvTQpHRWx1ZEdWeWJtVjBJRmRwWkdkcGRITWdVSFI1SUV4MFpEQWVGdzB5TXpBM01URXhNelEyTURCYUZ3MHlOREEzCk1UQXhNelEyTURCYU1FVXhDekFKQmdOVkJBWVRBa0ZWTVJNd0VRWURWUVFJREFwVGIyMWxMVk4wWVhSbE1TRXcKSHdZRFZRUUtEQmhKYm5SbGNtNWxkQ0JYYVdSbmFYUnpJRkIwZVNCTWRHUXdnZ0VpTUEwR0NTcUdTSWIzRFFFQgpBUVVBQTRJQkR3QXdnZ0VLQW9JQkFRQzNDa3ZEMEtmdXh0V2g0UUdXVzFrbnhQeldiU2Jlako5M29BRk9XYkw1CnFvM01FT0FJdkl4NFVndEcvelk3dW03aEtiZnh0OVg0OEVDT3UwUW9CZUt6TjRDZTYvQmFJWGtidjBoeXUwSE0KVy9mQ1JUbFhPWDIrcFIvTm9HTVZ4QW5FL3hRL0dmUUlMRlVJZDZ0eTBBL20zTHZzTE5BL1RxRDAzVTNrMVk3MgowdmZJWU1ZZkhPR3dSOGFGSFB4OEhINkRYbGFVV21lQjJZZVExdkVpNVo3OE41eXFlTHgrOWdBeEwxZ2dBYlVnClg2RWFCcnBpWE9ndFF5NFhRVzJ2a0J0Z1RtMUc3eFhGYks3UVJYZGlrV1U3QnA2Tjhield5YzdJZS9YTlRKbTEKOVRRSlBXOC9sTGlrZ21mUFZSQ0ZNZisvQzJwQjhlNkE4S0F6dVQvbmtLdmZBZ01CQUFHalV6QlJNQjBHQTFVZApEZ1FXQkJTd0ovV1VjeE5NOVZlbjhxL3d6eUN3bG15aFBUQWZCZ05WSFNNRUdEQVdnQlN3Si9XVWN4Tk05VmVuCjhxL3d6eUN3bG15aFBUQVBCZ05WSFJNQkFmOEVCVEFEQVFIL01BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQ2YKTVNOa01WZE1NT054ZDF1azZoU2lwRFpYSm4zMXN0UlBuZWRRQ0twQ2t0ZXN4RmpkanlTemdaWVRzZm5SZHR2TgpWVVVBa1RzOEZDa3ZJMS8wY21OKzlLNlRLZWR2WnNSMktjcGR3VUhDeHZEZEd3Um1Udnh2MDhvd3BNR2lGZm1tClZNYk5PZ2NNN0hrdy9McDliL1d5dmJaejNPTGNlbDJvb1VvaHRlSkhNcW96dlYyZkVkcExEcWdiNktzN1l4SDgKb2U0QWZyNXlNOTU5WDJ2b2g1Z3h4bHFIWjdoemJKWTlibml2bEJrN1preEdLREpSMVNKeG45WDd6RisybUljNgozQUlWYUtnUkt5SkE0QzRYN0VydnA5WG1WUzAvMDMrK09pOXFRZHVkV0VYc0YwNVN4N1NXTmp3amQ1cm1jOU9oCnMxdEFFNktFa2pObE9XdHVlZTZxCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K',
      })
    );
  });

  it('should show validation error if base64 private key is invalid', async () => {
    const { user } = setup({ settingsUpdated: mockUpdate });

    await user.type(screen.getByLabelText(/Private key/), 'WRONG');
    await user.click(screen.getByRole('button', { name: /^Next/i }));

    expect(screen.getByText(/^Not valid PEM key/i)).toBeInTheDocument();
  });

  it('should show validation error if base64 certificate is invalid', async () => {
    const { user } = setup({ settingsUpdated: mockUpdate });

    await user.type(screen.getByLabelText(/Certificate/), 'WRONG');
    await user.click(screen.getByRole('button', { name: /^Next/i }));

    expect(screen.getByText(/^Not valid PEM certificate/i)).toBeInTheDocument();
  });
});
