module mod_BaseLossFunction
implicit none
    
!-------------------
! 抽象类：损失函数 |
!-------------------
type, abstract, public :: BaseLossFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* 损失函数
    procedure(m_f), deferred, public :: f 
    !* 损失函数导数
    procedure(m_df), deferred, public :: df  

end type BaseLossFunction
!===================
    

!-------------------
! 抽象类：函数接口 |
!-------------------	
abstract interface   

	!* 损失函数
	subroutine m_f( this, t, y, ans )
    use mod_Precision
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t 是目标输出向量，y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans

	end subroutine
	!====
	
	!* 损失函数一阶导数
	!* 返回对网络预测向量的导数
	subroutine m_df( this, t, y, dy )
    use mod_Precision
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t 是目标输出向量，y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), dimension(:), intent(inout) :: dy

	end subroutine
	!====

end interface
!===================
    
end module