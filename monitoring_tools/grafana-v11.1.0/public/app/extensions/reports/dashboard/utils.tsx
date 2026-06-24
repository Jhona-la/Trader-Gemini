import React from 'react';

import { SceneGridItemLike, SceneGridRow } from '@grafana/scenes';
import { GRID_CELL_HEIGHT, GRID_CELL_VMARGIN, GRID_COLUMN_COUNT } from 'app/core/constants';

export interface ReportPage {
  items: ReportGridItem[];
  h: number;
  pageBreakAfter?: boolean;
}

export interface ReportGridItem {
  x: number;
  y: number;
  w: number;
  h: number;
  render: () => React.ReactNode;
}

export function buildSimpleLayout(
  children: SceneGridItemLike[],
  scaleFactor: number,
  screenHeight: number,
  isLandscape: boolean
): ReportPage[] {
  const pages: ReportPage[] = [];
  const maxRowCount = Math.floor(screenHeight / (GRID_CELL_HEIGHT + GRID_CELL_VMARGIN));
  let panelHeight = maxRowCount;
  if (!isLandscape) {
    panelHeight = maxRowCount / 3;
  }
  let yPos = 0;

  let currentPage: ReportPage = {
    items: [],
    h: 0,
  };
  pages.push(currentPage);

  let rowCount = 0;
  let isAfterRow = false;

  for (let i = 0; i < children.length; i++) {
    const gridChild = children[i];
    const isRow = gridChild instanceof SceneGridRow;
    const blockItem: ReportGridItem = {
      x: 0,
      y: yPos,
      w: GRID_COLUMN_COUNT,
      h: isRow ? scaleFactor : isAfterRow ? panelHeight - scaleFactor : panelHeight,
      render: () => <gridChild.Component model={gridChild} key={gridChild.state.key} />,
    };

    if (isRow) {
      // Get row children
      children.splice(i + 1, 0, ...gridChild.state.children);
      rowCount++;
    }

    if (yPos >= maxRowCount) {
      currentPage.pageBreakAfter = true;
      yPos = 0;
      blockItem.y = 0;

      currentPage = {
        items: [],
        h: 0,
      };

      pages.push(currentPage);
    }

    yPos += blockItem.h;

    currentPage.items.push(blockItem);
    currentPage.h = blockItem.y + blockItem.h;

    isAfterRow = isRow;
  }

  return pages;
}

export function buildGridLayout(
  children: SceneGridItemLike[],
  scaleFactor: number,
  screenHeight: number
): ReportPage[] {
  const pages: ReportPage[] = [];
  const maxRowCount = Math.floor(screenHeight / (GRID_CELL_HEIGHT + GRID_CELL_VMARGIN));
  let yShift = 0;
  let isAfterRow = false;

  let currentPage: ReportPage = {
    items: [],
    h: 0,
  };
  pages.push(currentPage);

  for (let i = 0; i < children.length; i++) {
    const gridChild = children[i];
    const isRow = gridChild instanceof SceneGridRow;
    const blockItem = getBlockItem(gridChild, yShift, maxRowCount, scaleFactor, isAfterRow);

    if (isRow) {
      // Get row children
      children.splice(i + 1, 0, ...gridChild.state.children);
    }

    const nextBlockItem =
      i < children.length - 1 ? getBlockItem(children[i + 1], yShift, maxRowCount, scaleFactor, isRow) : null;

    // Handle panel overflow (or next panel overflow if current panel is a row)
    let newH = Math.max(currentPage.h, blockItem.y + blockItem.h);
    let nextPanelH = 0;

    if (isRow && nextBlockItem != null) {
      nextPanelH = Math.max(currentPage.h, nextBlockItem.y - scaleFactor + 1 + nextBlockItem.h);
    }

    if (newH > maxRowCount || (isRow && nextPanelH > maxRowCount)) {
      currentPage.pageBreakAfter = true;
      yShift = gridChild.state.y! * scaleFactor;
      blockItem.y = 0;

      currentPage = {
        items: [],
        h: 0,
      };

      pages.push(currentPage);
    }

    currentPage.items.push(blockItem);
    currentPage.h = Math.max(currentPage.h, blockItem.y + blockItem.h);

    if (isRow) {
      isAfterRow = true;
      yShift += scaleFactor - 1;
    } else if (nextBlockItem != null && nextBlockItem.y !== blockItem.y) {
      isAfterRow = false;
    }
  }

  return pages;
}

function getBlockItem(
  gridChild: SceneGridItemLike,
  yShift: number,
  maxRowCount: number,
  scaleFactor: number,
  isAfterRow: boolean
): ReportGridItem {
  const blockItem: ReportGridItem = {
    x: gridChild.state.x!,
    y: gridChild.state.y! * scaleFactor - yShift,
    w: gridChild.state.width!,
    h: gridChild.state.height! * scaleFactor,
    render: () => <gridChild.Component model={gridChild} key={gridChild.state.key} />,
  };

  // If the panel is a row, its height should be 1, no matter the scale factor
  if (gridChild instanceof SceneGridRow) {
    blockItem.h = 1;
  }

  // If the panel is bigger than the maximum number of rows we can display in a page
  if (blockItem.h > maxRowCount) {
    if (isAfterRow) {
      blockItem.h = maxRowCount - 1;
    } else {
      blockItem.h = maxRowCount;
    }
  }

  return blockItem;
}

export type Position = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type PositionParams = {
  margin: [number, number];
  containerPadding: [number, number];
  containerWidth: number;
  cols: number;
  rowHeight: number;
  maxRows: number;
};

export function getGridParams(screenWidth: number): PositionParams {
  return {
    margin: [GRID_CELL_VMARGIN, GRID_CELL_VMARGIN],
    containerWidth: screenWidth,
    containerPadding: [0, 0],
    cols: GRID_COLUMN_COUNT,
    rowHeight: GRID_CELL_HEIGHT,
    maxRows: 10000,
  };
}

// Helper for generating column width
export function calcGridColWidth(positionParams: PositionParams): number {
  const { margin, containerPadding, containerWidth, cols } = positionParams;
  return (containerWidth - margin[0] * (cols - 1) - containerPadding[0] * 2) / cols;
}

// This can either be called:
// calcGridItemWHPx(w, colWidth, margin[0])
// or
// calcGridItemWHPx(h, rowHeight, margin[1])
export function calcGridItemWHPx(gridUnits: number, colOrRowSize: number, marginPx: number): number {
  // 0 * Infinity === NaN, which causes problems with resize contraints
  if (!Number.isFinite(gridUnits)) {
    return gridUnits;
  }
  return Math.round(colOrRowSize * gridUnits + Math.max(0, gridUnits - 1) * marginPx);
}

/**
 * Return position on the page given an x, y, w, h.
 * left, top, width, height are all in pixels.
 * @param  {PositionParams} positionParams  Parameters of grid needed for coordinates calculations.
 * @param  {Number}  x                      X coordinate in grid units.
 * @param  {Number}  y                      Y coordinate in grid units.
 * @param  {Number}  w                      W coordinate in grid units.
 * @param  {Number}  h                      H coordinate in grid units.
 * @return {Position}                       Object containing coords.
 */
export function calcGridItemPosition(
  positionParams: PositionParams,
  x: number,
  y: number,
  w: number,
  h: number
): Position {
  const { margin, containerPadding, rowHeight } = positionParams;
  const colWidth = calcGridColWidth(positionParams);
  const out: Position = {
    width: calcGridItemWHPx(w, colWidth, margin[0]),
    height: calcGridItemWHPx(h, rowHeight, margin[1]),
    top: Math.round((rowHeight + margin[1]) * y + containerPadding[1]),
    left: Math.round((colWidth + margin[0]) * x + containerPadding[0]),
  };

  return out;
}

/**
 * Translate x and y coordinates from pixels to grid units.
 * @param  {PositionParams} positionParams  Parameters of grid needed for coordinates calculations.
 * @param  {Number} top                     Top position (relative to parent) in pixels.
 * @param  {Number} left                    Left position (relative to parent) in pixels.
 * @param  {Number} w                       W coordinate in grid units.
 * @param  {Number} h                       H coordinate in grid units.
 * @return {Object}                         x and y in grid units.
 */
export function calcXY(
  positionParams: PositionParams,
  top: number,
  left: number,
  w: number,
  h: number
): { x: number; y: number } {
  const { margin, cols, rowHeight, maxRows } = positionParams;
  const colWidth = calcGridColWidth(positionParams);

  // left = colWidth * x + margin * (x + 1)
  // l = cx + m(x+1)
  // l = cx + mx + m
  // l - m = cx + mx
  // l - m = x(c + m)
  // (l - m) / (c + m) = x
  // x = (left - margin) / (coldWidth + margin)
  let x = Math.round((left - margin[0]) / (colWidth + margin[0]));
  let y = Math.round((top - margin[1]) / (rowHeight + margin[1]));

  // Capping
  x = clamp(x, 0, cols - w);
  y = clamp(y, 0, maxRows - h);
  return { x, y };
}

/**
 * Given a height and width in pixel values, calculate grid units.
 * @param  {PositionParams} positionParams  Parameters of grid needed for coordinates calcluations.
 * @param  {Number} height                  Height in pixels.
 * @param  {Number} width                   Width in pixels.
 * @param  {Number} x                       X coordinate in grid units.
 * @param  {Number} y                       Y coordinate in grid units.
 * @param {String} handle Resize Handle.
 * @return {Object}                         w, h as grid units.
 */
export function calcWH(
  positionParams: PositionParams,
  width: number,
  height: number,
  x: number,
  y: number,
  handle: string
): { w: number; h: number } {
  const { margin, maxRows, cols, rowHeight } = positionParams;
  const colWidth = calcGridColWidth(positionParams);

  // width = colWidth * w - (margin * (w - 1))
  // ...
  // w = (width + margin) / (colWidth + margin)
  let w = Math.round((width + margin[0]) / (colWidth + margin[0]));
  let h = Math.round((height + margin[1]) / (rowHeight + margin[1]));

  // Capping
  let _w = clamp(w, 0, cols - x);
  let _h = clamp(h, 0, maxRows - y);
  if (['sw', 'w', 'nw'].indexOf(handle) !== -1) {
    _w = clamp(w, 0, cols);
  }
  if (['nw', 'n', 'ne'].indexOf(handle) !== -1) {
    _h = clamp(h, 0, maxRows);
  }
  return { w: _w, h: _h };
}

// Similar to _.clamp
export function clamp(num: number, lowerBound: number, upperBound: number): number {
  return Math.max(Math.min(num, upperBound), lowerBound);
}
