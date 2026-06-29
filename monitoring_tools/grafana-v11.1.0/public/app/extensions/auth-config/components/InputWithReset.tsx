import { css } from '@emotion/css';
import React from 'react';

import { GrafanaTheme2 } from '@grafana/data';
import { Button, Input, useStyles2 } from '@grafana/ui';

export type Props = React.ComponentProps<typeof Input> & {
  isConfigured: boolean;
  onReset: () => void;
};

export const InputWithReset = ({ isConfigured, onReset, ...props }: Props): JSX.Element => {
  const styles = useStyles2(getStyles);

  return (
    <div className={styles.inputWithReset}>
      {isConfigured && <Input {...props} value="configured" disabled />}
      {!isConfigured && <Input {...props} />}

      {isConfigured && (
        <Button className={styles.resetButton} variant="primary" size="md" onClick={onReset}>
          Reset
        </Button>
      )}
    </div>
  );
};

const getStyles = (theme: GrafanaTheme2) => {
  return {
    resetButton: css`
      border-top-left-radius: 0px;
      border-bottom-left-radius: 0px;
    `,
    inputWithReset: css`
      padding: 0;
      display: flex;
    `,
  };
};
