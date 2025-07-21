"use client";

import React from "react";
import { Card, CardContent } from "@/components/ui/card";

interface SeatMapProps {
    onSeatSelect: (seatNumber: string) => void;
    selectedSeat?: string;
}

// 定义典型窄体飞机的座位布局
const SEAT_LAYOUT = {
    business: { rows: [1, 2, 3, 4], seatsPerRow: ['A', 'B', 'C', 'D'] },
    economyPlus: { rows: [5, 6, 7, 8], seatsPerRow: ['A', 'B', 'C', 'D', 'E', 'F'] },
    economy: {
        rows: Array.from({ length: 16 }, (_, i) => i + 9), // 9-24排
        seatsPerRow: ['A', 'B', 'C', 'D', 'E', 'F']
    }
};

const OCCUPIED_SEATS = new Set([
    '1A', '2B', '3C', '5A', '5F', '7B', '7E', '9A', '9F', '10C', '10D',
    '12A', '12F', '14B', '14E', '16A', '16F', '18C', '18D', '20A', '20F',
    '22B', '22E', '24A', '24F'
]);

const EXIT_ROWS = new Set([4, 16]);

export function SeatMap({ onSeatSelect, selectedSeat }: SeatMapProps) {
    const getSeatStatus = (seatNumber: string) => {
        if (OCCUPIED_SEATS.has(seatNumber)) return 'occupied';
        if (selectedSeat === seatNumber) return 'selected';
        return 'available';
    };

    const getSeatColor = (status: string, isExit: boolean) => {
        // 可用 = 绿色, 已占用 = 灰色, 安全出口排 = 黄色（柔和）
        switch (status) {
            case 'occupied':
                return 'bg-gray-300 text-gray-500 cursor-not-allowed';
            case 'selected':
                return 'bg-emerald-600 text-white cursor-pointer hover:bg-emerald-700';
            case 'available':
                return isExit
                    ? 'bg-yellow-100 hover:bg-yellow-200 cursor-pointer border-yellow-300'
                    : 'bg-emerald-100 hover:bg-emerald-200 cursor-pointer border-emerald-300';
            default:
                return 'bg-emerald-100';
        }
    };

    const renderSeatSection = (title: string, config: typeof SEAT_LAYOUT.business, className: string) => (
        <div className={`mb-6 ${className}`}>
            <h4 className="text-sm font-semibold mb-2 text-center">{title}</h4>
            <div className="space-y-1">
                {config.rows.map(row => {
                    const isExitRow = EXIT_ROWS.has(row);
                    return (
                        <div key={row} className="flex items-center justify-center gap-1">
                            <span className="w-6 text-xs text-gray-500 text-right mr-2">{row}</span>
                            <div className="flex gap-1">
                                {config.seatsPerRow.slice(0, Math.ceil(config.seatsPerRow.length / 2)).map(letter => {
                                    const seatNumber = `${row}${letter}`;
                                    const status = getSeatStatus(seatNumber);
                                    return (
                                        <button
                                            key={seatNumber}
                                            className={`w-8 h-8 text-xs font-medium border rounded ${getSeatColor(status, isExitRow)} transition-colors`}
                                            onClick={() => status === 'available' && onSeatSelect(seatNumber)}
                                            disabled={status === 'occupied'}
                                            title={`座位 ${seatNumber}${isExitRow ? ' (安全出口排)' : ''}${status === 'occupied' ? ' - 已占用' : ''}`}
                                        >
                                            {letter}
                                        </button>
                                    );
                                })}
                            </div>
                            <div className="w-4" /> {/* 过道 */}
                            <div className="flex gap-1">
                                {config.seatsPerRow.slice(Math.ceil(config.seatsPerRow.length / 2)).map(letter => {
                                    const seatNumber = `${row}${letter}`;
                                    const status = getSeatStatus(seatNumber);
                                    return (
                                        <button
                                            key={seatNumber}
                                            className={`w-8 h-8 text-xs font-medium border rounded ${getSeatColor(status, isExitRow)} transition-colors`}
                                            onClick={() => status === 'available' && onSeatSelect(seatNumber)}
                                            disabled={status === 'occupied'}
                                            title={`座位 ${seatNumber}${isExitRow ? ' (安全出口排)' : ''}${status === 'occupied' ? ' - 已占用' : ''}`}
                                        >
                                            {letter}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );

    return (
        <Card className="w-full max-w-md mx-auto my-4 bg-blue-50">
            <CardContent className="p-4">
                <div className="text-center mb-4">
                    <h3 className="font-semibold text-lg mb-2">选择您的座位</h3>
                    <div className="flex justify-center gap-4 text-xs">
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 bg-emerald-100 border border-emerald-300 rounded"></div>
                            <span>可选</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 bg-gray-300 rounded"></div>
                            <span>已占用</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 bg-yellow-100 border border-yellow-300 rounded"></div>
                            <span>安全出口排</span>
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    {renderSeatSection("商务舱", SEAT_LAYOUT.business, "border-b pb-4")}
                    {renderSeatSection("经济舱Plus", SEAT_LAYOUT.economyPlus, "border-b pb-4")}
                    {renderSeatSection("经济舱", SEAT_LAYOUT.economy, "")}
                </div>

                {selectedSeat && (
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg text-center">
                        <p className="text-sm font-medium text-blue-800">
                            已选择: 座位 {selectedSeat}
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
} 