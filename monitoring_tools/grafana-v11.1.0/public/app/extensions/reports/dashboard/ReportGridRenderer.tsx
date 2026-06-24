import React, { useEffect, useState, useMemo } from 'react';
import { useLocation } from 'react-router';

import { SceneGridItemLike, SceneGridLayout, SceneGridRow, SceneTimeRangeState } from '@grafana/scenes';
import { GRID_COLUMN_COUNT } from 'app/core/constants';

import ReportFooter from './ReportFooter';
import ReportHeader from './ReportHeader';
import {
  buildSimpleLayout,
  buildGridLayout,
  calcGridItemPosition,
  getGridParams,
  PositionParams,
  ReportGridItem,
  ReportPage,
} from './utils';

const A4_WIDTH = 794;
const A4_HEIGHT = 1122;
const TOP_MARGIN = 130;
const SIDE_MARGIN = 16;

interface Props {
  grid: SceneGridLayout;
  title: string;
  timeRange: SceneTimeRangeState;
}

export function ReportGridRenderer({ grid, title, timeRange }: Props) {
  const [isActive, setIsActive] = useState(false);
  const { scaleFactor, isLandscape, isSimple } = useUrlValues();

  uncollapseRows(grid.state.children);
  const { children } = grid.useState();

  useEffect(() => {
    setIsActive(true);
    return grid.activate();
  }, [grid]);

  let pageWidth = A4_WIDTH;
  let pageHeight = A4_HEIGHT;
  if (isLandscape) {
    pageWidth = A4_HEIGHT;
    pageHeight = A4_WIDTH;
  }

  const screenHeight = (pageHeight - TOP_MARGIN) * scaleFactor;
  const screenWidth = pageWidth * scaleFactor - 2 * SIDE_MARGIN;

  const blocks: ReportPage[] = useMemo(() => {
    if (isSimple) {
      return buildSimpleLayout(children, scaleFactor, screenHeight, isLandscape);
    }

    return buildGridLayout(children, scaleFactor, screenHeight);
  }, [children, scaleFactor, screenHeight, isLandscape, isSimple]);

  if (!isActive) {
    return null;
  }

  return (
    <div style={{ position: 'relative' }}>
      {blocks.map((page, index) => (
        <div key={index}>
          <ReportHeader title={title} timeRange={timeRange} scaleFactor={scaleFactor} />
          <div style={getPageStyle(page, screenWidth)}>
            {page.items.map((item, index) => (
              <div key={index} style={getItemStyle(item, screenWidth)}>
                {item.render()}
              </div>
            ))}
          </div>
          <ReportFooter scaleFactor={scaleFactor} />
          {page.pageBreakAfter && (
            <>
              <div style={{ pageBreakAfter: 'always', marginBottom: '16px' }} />
              <div style={{ marginBottom: '16px' }} />
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export function useUrlValues() {
  const location = useLocation();
  const urlParams = new URLSearchParams(location.search);
  const scaleFactor = parseFloat(urlParams.get('scale') ?? '');

  return {
    scaleFactor: isNaN(scaleFactor) ? 1 : scaleFactor,
    isLandscape: urlParams.get('pdf.landscape') !== 'false',
    isSimple: urlParams.get('pdf.layout') === 'simple',
  };
}

function uncollapseRows(children: SceneGridItemLike[]) {
  for (const gridChild of children) {
    if (gridChild instanceof SceneGridRow && gridChild.state.isCollapsed) {
      gridChild.onCollapseToggle();
    }
  }
}

function getItemStyle(item: ReportGridItem, screenWidth: number): React.CSSProperties {
  const params: PositionParams = getGridParams(screenWidth);
  const position = calcGridItemPosition(params, item.x, item.y, item.w, item.h);

  return {
    top: position.top,
    left: position.left,
    width: position.width,
    height: position.height,
    position: 'absolute',
  };
}

function getPageStyle(block: ReportPage, screenWidth: number): React.CSSProperties {
  const params: PositionParams = getGridParams(screenWidth);
  const position = calcGridItemPosition(params, 0, 0, GRID_COLUMN_COUNT, block.h);

  return {
    width: position.width,
    height: position.height,
    position: 'relative',
  };
}
